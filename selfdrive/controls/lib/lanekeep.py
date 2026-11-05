from collections import deque
from enum import IntEnum
import numpy as np

from openpilot.common.constants import CV

DT_CTRL = 0.01


class LaneKeepAlert(IntEnum):
  NONE = 0
  KEEP_HANDS_ON_WHEEL = 1
  EMERGENCY_DEPARTURE = 2


# Default vehicle geometry for 2025 Subaru Legacy
DEFAULT_VEHICLE_HALF_WIDTH = 0.92  # meters (centerline to outer edge of tire)
DEFAULT_CAMERA_OFFSET = 0.04       # meters

# Standard Painted-Line Trigger Thresholds (~3 in margin from painted line)
STANDARD_LINE_MARGIN_M = 0.07
STANDARD_TTC_PROX_M = 0.10
STANDARD_TTC_THRESH_S = 0.75
DEFAULT_GRADIENT_WIDTH = 0.30

# Dynamics and Acceleration Limits
MAX_LAT_ACCEL = 0.40                       # m/s^2 gentle boundary-parallel arrest ceiling
OVERRIDE_TORQUE_THRESHOLD = 0.5            # Nm (used only when torque_unit_nm=True)
OVERRIDE_OPPOSING_CAN_TORQUE = 140.0       # Raw CAN torque opposing the nudge (above passive hand drag)
OVERRIDE_ASSISTING_CAN_TORQUE = 185.0      # Raw CAN torque steering with the nudge / holding into curve
OVERRIDE_HOLD_CAN_TORQUE = 90.0            # Raw CAN torque keeping override cooldown latched
OVERRIDE_CONFIRM_TIME_S = 0.05             # 50 ms debounce for CAN torque override
OVERRIDE_COOLDOWN_S = 1.20                 # Latched cooldown after override to prevent ON-OFF-ON pulsing
POST_ARREST_STRAIGHTEN_CAN_TORQUE = 60.0   # Gentle opposing CAN torque releasing hold once drift stops
POST_ARREST_INWARD_CAN_TORQUE = 80.0       # Inward recovery CAN torque releasing hold once drift slows
POST_ARREST_CONFIRM_TIME_S = 0.04          # 40 ms debounce for post-arrest release
POST_ARREST_COOLDOWN_S = 0.50              # Brief cooldown after driver takes over from an arrested drift
CORNERING_GATE_ANGLE_DEG = 45.0            # degrees (suppress assist during active turns)
APEX_CUT_BUFFER_M = 0.12                   # meters (reduce trigger margin on inside curve apex)
POST_BLINKER_COOLDOWN_S = 2.0              # seconds to wait after turn signal auto-cancels

# Rolling 30-Second Anti-Lazy Hands-Off Timeout (5s warning / 15s cutoff)
HANDS_OFF_WINDOW_S = 30.0
HANDS_OFF_WARN_S = 5.0
HANDS_OFF_LOCKOUT_S = 15.0
HANDS_OFF_GRACE_S = HANDS_OFF_LOCKOUT_S - HANDS_OFF_WARN_S  # 10.0s grace after 5.0s warning = 15.0s cutoff
HANDS_ON_IDLE_CAN_TORQUE = 85.0
HANDS_ON_ASSISTING_CAN_TORQUE = 115.0
HANDS_ON_CONFIRM_TIME_S = 0.12

# S-Curve & 20 Hz Vision Velocity Smoothing Filters
TAU_SMOOTH = 0.24
TAU_RELEASE = 0.08
TAU_OVERRIDE_RELEASE = 0.04
TAU_VEL = 0.16
MAX_D_DOT_MPS = 1.5
MIN_TTC_CLOSING_SPEED_MPS = 0.12
TTC_CONFIRM_TIME_S = 0.08
NUDGE_RELEASE_HYSTERESIS_M = 0.02
ACTIVE_CURVATURE_EPS = 1.0e-4
YELLOW_LINE_HOLD_S = 1.05


def _estimate_boundary_curvature(line) -> float:
  """Estimate spatial road curvature (1/m, +right / -left) from a modelV2 laneLine."""
  if line is None:
    return 0.0
  xs = getattr(line, "x", None)
  ys = getattr(line, "y", None)
  if xs is None or ys is None or len(xs) < 4 or len(ys) < 4:
    return 0.0
  x_arr = np.asarray(xs, dtype=np.float64)
  y_arr = np.asarray(ys, dtype=np.float64)
  n = min(len(x_arr), len(y_arr))
  x_arr = x_arr[:n]
  y_arr = y_arr[:n]
  mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr >= 2.0) & (x_arr <= 26.0)
  if int(np.count_nonzero(mask)) < 4:
    return 0.0
  xm = x_arr[mask]
  ym = y_arr[mask]
  if float(xm[-1] - xm[0]) < 8.0:
    return 0.0
  c2 = float(np.polyfit(xm, ym, 2)[0])
  return float(np.clip(2.0 * c2, -0.02, 0.02))


class LaneKeepController:
  def __init__(self, half_width: float = DEFAULT_VEHICLE_HALF_WIDTH, camera_offset: float = DEFAULT_CAMERA_OFFSET):
    self.w_half = half_width
    self.camera_offset = camera_offset

    # State variables
    self.active = False
    self.nudge_active = False
    self.desired_curvature = 0.0
    self.boundary_curvature = 0.0
    self.desired_lat_accel = 0.0
    self.left_line_state = 0   # 0: Off, 1: White, 3: Amber
    self.right_line_state = 0  # 0: Off, 1: White, 3: Amber
    self.alert = LaneKeepAlert.NONE

    # UI Feedback properties
    self.gradient_opacity = 0.0
    self.gradient_width = DEFAULT_GRADIENT_WIDTH

    # Internal tracking
    self._prev_curvature = 0.0
    self._prev_d_left = None
    self._prev_d_right = None
    self._d_dot_left = 0.0
    self._d_dot_right = 0.0
    self._left_line_curv = 0.0
    self._right_line_curv = 0.0
    self._curv_initialized = False
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._override_confirm_timer = 0.0
    self._straighten_confirm_timer = 0.0
    self._override_cooldown_left = 0.0
    self._override_cooldown_right = 0.0
    self._left_yellow_hold = 0.0
    self._right_yellow_hold = 0.0
    self._nudge_side = None
    self._episode_timer = 0.0
    self._window_samples: deque[tuple[float, float]] = deque()
    self._window_total_dt = 0.0
    self._window_active_dt = 0.0
    self._hands_off_warned = False
    self._hands_off_grace_timer = 0.0
    self._hands_on_confirm_timer = 0.0
    self._lockout = False
    self._blinker_cooldown_left = 0.0
    self._blinker_cooldown_right = 0.0
    self._speed_armed = False

  def _clear_hands_off_tracking(self):
    self._window_samples.clear()
    self._window_total_dt = 0.0
    self._window_active_dt = 0.0
    self._hands_off_warned = False
    self._hands_off_grace_timer = 0.0
    self._hands_on_confirm_timer = 0.0
    self._lockout = False

  def _record_window_sample(self, dt: float, active_dt: float):
    step_dt = max(0.0, float(dt))
    act_dt = max(0.0, float(active_dt))
    if step_dt <= 0.0:
      return
    self._window_samples.append((step_dt, act_dt))
    self._window_total_dt += step_dt
    self._window_active_dt += act_dt

    while self._window_samples and self._window_total_dt > HANDS_OFF_WINDOW_S + 1e-9:
      excess = self._window_total_dt - HANDS_OFF_WINDOW_S
      old_dt, old_act = self._window_samples[0]
      if old_dt <= excess + 1e-9:
        self._window_samples.popleft()
        self._window_total_dt -= old_dt
        self._window_active_dt -= old_act
      else:
        frac = excess / old_dt
        trim_act = old_act * frac
        self._window_samples[0] = (old_dt - excess, old_act - trim_act)
        self._window_total_dt -= excess
        self._window_active_dt -= trim_act
        break

    if self._window_active_dt < 1e-9:
      self._window_active_dt = 0.0

  def reset(self, full: bool = False):
    self.active = False
    self.nudge_active = False
    self.desired_curvature = 0.0
    self.boundary_curvature = 0.0
    self.desired_lat_accel = 0.0
    self.left_line_state = 0
    self.right_line_state = 0
    self.alert = LaneKeepAlert.NONE
    self.gradient_opacity = 0.0
    self.gradient_width = DEFAULT_GRADIENT_WIDTH
    self._prev_curvature = 0.0
    self._prev_d_left = None
    self._prev_d_right = None
    self._d_dot_left = 0.0
    self._d_dot_right = 0.0
    self._left_line_curv = 0.0
    self._right_line_curv = 0.0
    self._curv_initialized = False
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._override_confirm_timer = 0.0
    self._straighten_confirm_timer = 0.0
    self._override_cooldown_left = 0.0
    self._override_cooldown_right = 0.0
    self._left_yellow_hold = 0.0
    self._right_yellow_hold = 0.0
    self._nudge_side = None
    self._episode_timer = 0.0
    self._clear_hands_off_tracking()
    if full:
      self._speed_armed = False
      self._blinker_cooldown_left = 0.0
      self._blinker_cooldown_right = 0.0

  def update(
    self,
    model_v2,
    v_ego: float,
    steering_angle_deg: float,
    steering_torque: float,
    left_blinker: bool = False,
    right_blinker: bool = False,
    lat_active: bool = False,
    steering_pressed: bool = False,
    enabled: bool = True,
    min_activation_speed_mph: float = 25.0,
    dt: float = DT_CTRL,
    model_updated: bool = True,
    torque_unit_nm: bool = False,
  ):
    """
    Core painted-line Lane Keep Assist cycle executed at controls rate (100 Hz / DT_CTRL).
    Openpilot curvature sign convention:
      +curvature (> 0) = RIGHT turn (negative steeringAngleDeg)
      -curvature (< 0) = LEFT turn  (positive steeringAngleDeg)
    """
    # 1. Master enable check
    if not enabled:
      self.reset()
      return

    # 2. Hierarchy Check: If active centering (MADS/AOL) is driving, LKA yields full authority
    if lat_active:
      self.reset()
      return

    # 3. Operating Speed Window with Hysteresis
    v_activate = min_activation_speed_mph * CV.MPH_TO_MS
    v_deactivate = max(v_activate - (4.0 * CV.MPH_TO_MS), 12.0 * CV.MPH_TO_MS)

    if not self._speed_armed:
      if v_ego >= v_activate:
        self._speed_armed = True
    else:
      if v_ego < v_deactivate:
        self._speed_armed = False

    if not self._speed_armed:
      self.reset()
      return

    # 4. Turn Signal Handlers & Cooldown Timers
    if left_blinker:
      self._blinker_cooldown_left = POST_BLINKER_COOLDOWN_S
    elif self._blinker_cooldown_left > 0.0:
      self._blinker_cooldown_left = max(0.0, self._blinker_cooldown_left - dt)

    if right_blinker:
      self._blinker_cooldown_right = POST_BLINKER_COOLDOWN_S
    elif self._blinker_cooldown_right > 0.0:
      self._blinker_cooldown_right = max(0.0, self._blinker_cooldown_right - dt)

    # 5. Cornering Suppression Gate (Physical wheel turned >= 45 deg)
    steer_suppressed = abs(steering_angle_deg) >= CORNERING_GATE_ANGLE_DEG

    # 6. Standard Painted-Line Thresholds (~3 in from painted line, tight ~4 in drift lookahead)
    d_thresh = STANDARD_LINE_MARGIN_M
    ttc_line_prox = STANDARD_TTC_PROX_M
    ttc_thresh = STANDARD_TTC_THRESH_S
    self.gradient_width = DEFAULT_GRADIENT_WIDTH

    # 7. Extract Painted Lane-Line Vision Data
    left_prob = float(model_v2.laneLineProbs[1]) if len(model_v2.laneLineProbs) > 1 else 0.0
    right_prob = float(model_v2.laneLineProbs[2]) if len(model_v2.laneLineProbs) > 2 else 0.0
    left_line_visible = left_prob >= 0.40
    right_line_visible = right_prob >= 0.40

    left_y = float(model_v2.laneLines[1].y[0]) if len(model_v2.laneLines) > 1 and len(model_v2.laneLines[1].y) > 0 else -1.8
    right_y = float(model_v2.laneLines[2].y[0]) if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[2].y) > 0 else 1.8

    if model_updated or not self._curv_initialized:
      self._curv_initialized = True
      self._left_line_curv = _estimate_boundary_curvature(model_v2.laneLines[1]) if (left_line_visible and len(model_v2.laneLines) > 1) else 0.0
      self._right_line_curv = _estimate_boundary_curvature(model_v2.laneLines[2]) if (right_line_visible and len(model_v2.laneLines) > 2) else 0.0

    # Distance from tire outer edges to painted lane lines
    d_left = -left_y - (self.w_half + self.camera_offset)
    d_right = right_y - (self.w_half - self.camera_offset)

    # Lateral velocities updated strictly on 20 Hz modelV2 frames with EMA filtering
    # (prevents 5x derivative spikes from differentiating 20 Hz vision at 100 Hz DT_CTRL)
    self._dt_since_model += max(0.0, dt)
    if model_updated and self._dt_since_model > 0.0:
      dt_m = max(self._dt_since_model, 1e-3)
      self._dt_since_model = 0.0
      alpha_v = float(dt_m / (TAU_VEL + dt_m))

      if left_line_visible:
        if self._prev_d_left is not None:
          raw_dl = float(np.clip((d_left - self._prev_d_left) / dt_m, -MAX_D_DOT_MPS, MAX_D_DOT_MPS))
          self._d_dot_left = (1.0 - alpha_v) * self._d_dot_left + alpha_v * raw_dl
        self._prev_d_left = d_left
      else:
        self._prev_d_left = None
        self._d_dot_left = 0.0

      if right_line_visible:
        if self._prev_d_right is not None:
          raw_dr = float(np.clip((d_right - self._prev_d_right) / dt_m, -MAX_D_DOT_MPS, MAX_D_DOT_MPS))
          self._d_dot_right = (1.0 - alpha_v) * self._d_dot_right + alpha_v * raw_dr
        self._prev_d_right = d_right
      else:
        self._prev_d_right = None
        self._d_dot_right = 0.0

    d_dot_left = self._d_dot_left
    d_dot_right = self._d_dot_right

    # Time-To-Cross (TTC) for painted lines
    ttc_left = max(0.0, d_left) / abs(d_dot_left) if d_dot_left < -MIN_TTC_CLOSING_SPEED_MPS else 99.0
    ttc_right = max(0.0, d_right) / abs(d_dot_right) if d_dot_right < -MIN_TTC_CLOSING_SPEED_MPS else 99.0

    # Curve Apex Allowance: reduce trigger margin on inside of curve (+road_curv = right curve, -road_curv = left curve)
    road_curv = float(model_v2.action.desiredCurvature) if hasattr(model_v2, "action") else 0.0
    d_thresh_left = max(-0.06, d_thresh - APEX_CUT_BUFFER_M) if road_curv < -0.001 else d_thresh
    d_thresh_right = max(-0.06, d_thresh - APEX_CUT_BUFFER_M) if road_curv > 0.001 else d_thresh
    ttc_prox_left = ttc_line_prox
    ttc_prox_right = ttc_line_prox
    if left_line_visible and right_line_visible:
      # Narrow-lane pinch guard: when lane width < ~2.45m, never trigger near lane center or ping-pong across a tight pinch
      half_clearance = 0.5 * (d_left + d_right)
      d_thresh_left = min(d_thresh_left, max(-0.04, half_clearance - 0.12))
      d_thresh_right = min(d_thresh_right, max(-0.04, half_clearance - 0.12))
      ttc_prox_left = min(ttc_line_prox, max(0.0, (half_clearance - 0.06) - d_thresh_left))
      ttc_prox_right = min(ttc_line_prox, max(0.0, (half_clearance - 0.06) - d_thresh_right))

    # Proximity-gated & debounced TTC triggers
    if left_line_visible and d_left < (d_thresh_left + ttc_prox_left) and ttc_left < ttc_thresh:
      self._ttc_left_timer += dt
    else:
      self._ttc_left_timer = 0.0

    if right_line_visible and d_right < (d_thresh_right + ttc_prox_right) and ttc_right < ttc_thresh:
      self._ttc_right_timer += dt
    else:
      self._ttc_right_timer = 0.0

    left_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "left" and d_dot_left < 0.03) else 0.0
    right_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "right" and d_dot_right < 0.03) else 0.0

    sustain_left_line = (
      (self._nudge_side == "left") and
      ((d_left < d_thresh_left + ttc_prox_left and d_dot_left < -0.015) or
       (d_left < d_thresh_left + 0.055 and d_dot_left < 0.015))
    )
    sustain_right_line = (
      (self._nudge_side == "right") and
      ((d_right < d_thresh_right + ttc_prox_right and d_dot_right < -0.015) or
       (d_right < d_thresh_right + 0.055 and d_dot_right < 0.015))
    )

    # 8. Candidate Trigger Detection (Left vs. Right painted lane lines before override cooldown gating)
    raw_trigger_left_line = (
      left_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_left <= 0.0 and
      (d_left < (d_thresh_left + left_hyst) or self._ttc_left_timer >= TTC_CONFIRM_TIME_S or sustain_left_line)
    )

    raw_trigger_right_line = (
      right_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_right <= 0.0 and
      (d_right < (d_thresh_right + right_hyst) or self._ttc_right_timer >= TTC_CONFIRM_TIME_S or sustain_right_line)
    )

    # 9. Directional Driver Override, Latched Cooldown & Hands-On Detection
    if self._override_cooldown_left > 0.0:
      holding_left = (abs(steering_torque) > 0.35) if torque_unit_nm else (steering_torque > OVERRIDE_HOLD_CAN_TORQUE)
      left_returned_safe = (not left_line_visible or d_left > d_thresh_left + 0.12)
      if holding_left:
        self._override_cooldown_left = max(self._override_cooldown_left - dt, 0.35)
      elif left_returned_safe:
        self._override_cooldown_left = 0.0
      else:
        self._override_cooldown_left = max(0.0, self._override_cooldown_left - dt)

    if self._override_cooldown_right > 0.0:
      holding_right = (abs(steering_torque) > 0.35) if torque_unit_nm else (steering_torque < -OVERRIDE_HOLD_CAN_TORQUE)
      right_returned_safe = (not right_line_visible or d_right > d_thresh_right + 0.12)
      if holding_right:
        self._override_cooldown_right = max(self._override_cooldown_right - dt, 0.35)
      elif right_returned_safe:
        self._override_cooldown_right = 0.0
      else:
        self._override_cooldown_right = max(0.0, self._override_cooldown_right - dt)

    if raw_trigger_left_line or self._nudge_side == "left" or self.desired_curvature > ACTIVE_CURVATURE_EPS:
      candidate_side = "left"
    elif raw_trigger_right_line or self._nudge_side == "right" or self.desired_curvature < -ACTIVE_CURVATURE_EPS:
      candidate_side = "right"
    else:
      candidate_side = None

    post_arrest_release = False
    if torque_unit_nm:
      self._override_confirm_timer = 0.0
      self._straighten_confirm_timer = 0.0
      driver_override = steering_pressed or abs(steering_torque) > OVERRIDE_TORQUE_THRESHOLD
    else:
      # Positive CAN steeringTorque = driver pushing LEFT; negative = driver pushing RIGHT.
      # Passive resting-hand inertia against an EPS angle nudge registers ~40-115 CAN units;
      # intentional opposing push exceeds 140 CAN units, while same-direction curve-holding
      # or recovery assist uses a higher 185 CAN unit ceiling.
      if candidate_side == "left":
        torque_exceeded = (steering_torque > OVERRIDE_OPPOSING_CAN_TORQUE) or (steering_torque < -OVERRIDE_ASSISTING_CAN_TORQUE)
      elif candidate_side == "right":
        torque_exceeded = (steering_torque < -OVERRIDE_OPPOSING_CAN_TORQUE) or (steering_torque > OVERRIDE_ASSISTING_CAN_TORQUE)
      else:
        torque_exceeded = steering_pressed or abs(steering_torque) > OVERRIDE_OPPOSING_CAN_TORQUE

      self._override_confirm_timer = (self._override_confirm_timer + dt) if torque_exceeded else 0.0
      if candidate_side is not None:
        driver_override = self._override_confirm_timer >= OVERRIDE_CONFIRM_TIME_S
      else:
        driver_override = torque_exceeded

      # Post-arrest release: once outward painted-line drift has been arrested or slowed,
      # allow either light straightening pressure (> 60 CAN units) or inward recovery pressure
      # (> 80 CAN units away from the boundary) to release the hold immediately without drag.
      if not driver_override and self.active and self._episode_timer >= 0.15:
        if (
          candidate_side == "left" and
          (d_dot_left >= 0.02 or (d_left >= d_thresh_left and d_dot_left >= -0.005)) and
          (steering_torque > POST_ARREST_STRAIGHTEN_CAN_TORQUE or steering_torque < -POST_ARREST_INWARD_CAN_TORQUE)
        ):
          self._straighten_confirm_timer += dt
        elif (
          candidate_side == "right" and
          (d_dot_right >= 0.02 or (d_right >= d_thresh_right and d_dot_right >= -0.005)) and
          (steering_torque < -POST_ARREST_STRAIGHTEN_CAN_TORQUE or steering_torque > POST_ARREST_INWARD_CAN_TORQUE)
        ):
          self._straighten_confirm_timer += dt
        else:
          self._straighten_confirm_timer = 0.0
        if self._straighten_confirm_timer >= POST_ARREST_CONFIRM_TIME_S:
          post_arrest_release = True
          self._straighten_confirm_timer = 0.0
      else:
        self._straighten_confirm_timer = 0.0

    if driver_override:
      if candidate_side == "left":
        self._override_cooldown_left = OVERRIDE_COOLDOWN_S
      elif candidate_side == "right":
        self._override_cooldown_right = OVERRIDE_COOLDOWN_S
    elif post_arrest_release:
      if candidate_side == "left":
        self._override_cooldown_left = max(self._override_cooldown_left, POST_ARREST_COOLDOWN_S)
      elif candidate_side == "right":
        self._override_cooldown_right = max(self._override_cooldown_right, POST_ARREST_COOLDOWN_S)

    # Confirm actual driver hands-on steering torque (distinct from passive wheel inertia while EPS moves,
    # and never satisfied merely by coasting back through lane center hands-off).
    if driver_override or post_arrest_release:
      self._hands_on_confirm_timer = 0.0
      driver_hands_on = True
    else:
      if torque_unit_nm:
        hands_torque = steering_pressed or abs(steering_torque) > OVERRIDE_TORQUE_THRESHOLD
      elif not self.active:
        hands_torque = steering_pressed or abs(steering_torque) >= HANDS_ON_IDLE_CAN_TORQUE
      else:
        hands_torque = (
          (candidate_side == "left" and steering_torque <= -HANDS_ON_ASSISTING_CAN_TORQUE) or
          (candidate_side == "right" and steering_torque >= HANDS_ON_ASSISTING_CAN_TORQUE)
        )
      if hands_torque:
        self._hands_on_confirm_timer += dt
      else:
        self._hands_on_confirm_timer = 0.0
      driver_hands_on = self._hands_on_confirm_timer >= HANDS_ON_CONFIRM_TIME_S

    if driver_hands_on and (self._hands_off_warned or self._lockout):
      self._clear_hands_off_tracking()

    trigger_left_line = raw_trigger_left_line and (self._override_cooldown_left <= 0.0)
    trigger_right_line = raw_trigger_right_line and (self._override_cooldown_right <= 0.0)

    # 10. Compute Drift-Stopping Boundary Guidance (Curvature)
    target_curvature = 0.0
    is_left_nudge = False
    is_right_nudge = False

    # Gentle drift-arresting gains: damps outward drift toward the line and fades out
    # as soon as outward drift stops (d_dot > 0) so the car parallels the line instead
    # of pushing back across the lane.
    kp = 0.0024
    kd = 0.0014

    if not driver_override and not post_arrest_release:
      if trigger_left_line:
        is_left_nudge = True
        raw_b_curv = self._left_line_curv
        if not self.active:
          self.boundary_curvature = raw_b_curv
        else:
          alpha_b = float(dt / (TAU_SMOOTH + dt))
          self.boundary_curvature = (1.0 - alpha_b) * self.boundary_curvature + alpha_b * raw_b_curv
        if not self._lockout:
          d_lim = d_thresh_left + left_hyst
          sustain_err = min(0.02, (d_thresh_left + 0.055) - d_left) if sustain_left_line else 0.0
          d_err = max(0.0, d_lim - d_left, sustain_err)
          vel_left = max(0.0, -d_dot_left)
          recede_scale = max(0.0, 1.0 - (d_dot_left / 0.05)) if d_dot_left > 0.0 else 1.0
          # Positive curvature turns RIGHT away from left lane line
          target_curvature = float((kp * d_err + kd * vel_left) * recede_scale)
      elif trigger_right_line:
        is_right_nudge = True
        raw_b_curv = self._right_line_curv
        if not self.active:
          self.boundary_curvature = raw_b_curv
        else:
          alpha_b = float(dt / (TAU_SMOOTH + dt))
          self.boundary_curvature = (1.0 - alpha_b) * self.boundary_curvature + alpha_b * raw_b_curv
        if not self._lockout:
          d_lim = d_thresh_right + right_hyst
          sustain_err = min(0.02, (d_thresh_right + 0.055) - d_right) if sustain_right_line else 0.0
          d_err = max(0.0, d_lim - d_right, sustain_err)
          vel_right = max(0.0, -d_dot_right)
          recede_scale = max(0.0, 1.0 - (d_dot_right / 0.05)) if d_dot_right > 0.0 else 1.0
          # Negative curvature turns LEFT away from right lane line
          target_curvature = -float((kp * d_err + kd * vel_right) * recede_scale)

    actively_nudging = (is_left_nudge or is_right_nudge) and abs(target_curvature) > 0.0
    if is_left_nudge and abs(target_curvature) > 0.0:
      self._nudge_side = "left"
    elif is_right_nudge and abs(target_curvature) > 0.0:
      self._nudge_side = "right"
    else:
      self._nudge_side = None

    # Clamp lateral acceleration to 0.40 m/s^2 ceiling
    max_curvature = MAX_LAT_ACCEL / max(v_ego * v_ego, 1.0)
    target_curvature = float(np.clip(target_curvature, -max_curvature, max_curvature))

    # 11. Rolling 30-Second Anti-Lazy Hands-Off Timeout (5s Warning / 15s Cutoff)
    nudge_triggered = is_left_nudge or is_right_nudge
    override_active = driver_override or post_arrest_release or (self._override_cooldown_left > 0.0) or (self._override_cooldown_right > 0.0)
    active_push_step = actively_nudging and not override_active and not self._lockout

    if active_push_step or self.active:
      self._episode_timer += dt
    else:
      self._episode_timer = 0.0

    self._record_window_sample(dt, dt if active_push_step else 0.0)

    if not self._lockout:
      if not self._hands_off_warned:
        if self._window_active_dt >= HANDS_OFF_WARN_S - 1e-6:
          self._hands_off_warned = True
          # Include any overshoot past HANDS_OFF_WARN_S in the grace timer so large-dt steps hit 15.0s exact
          self._hands_off_grace_timer = max(0.0, self._window_active_dt - HANDS_OFF_WARN_S)
      else:
        if self._window_active_dt <= 0.0 and not nudge_triggered:
          self._hands_off_warned = False
          self._hands_off_grace_timer = 0.0
        else:
          self._hands_off_grace_timer += dt

      if self._hands_off_warned and (
        self._window_active_dt >= HANDS_OFF_LOCKOUT_S - 1e-6 or
        (self._hands_off_grace_timer >= HANDS_OFF_GRACE_S - 1e-6 and nudge_triggered)
      ):
        self._lockout = True

    if self._lockout:
      target_curvature = 0.0
      self._nudge_side = None
      actively_nudging = False
      self.alert = LaneKeepAlert.EMERGENCY_DEPARTURE
    elif self._hands_off_warned:
      self.alert = LaneKeepAlert.KEEP_HANDS_ON_WHEEL
    else:
      self.alert = LaneKeepAlert.NONE

    self.nudge_active = bool(actively_nudging and not override_active and not self._lockout)

    # 12. Asymmetric S-Curve Envelope Smoothing Filter (gentle onset, crisp release, fast yield on override)
    if override_active:
      tau = TAU_OVERRIDE_RELEASE
    elif abs(target_curvature) > abs(self._prev_curvature) and (target_curvature * self._prev_curvature >= 0.0):
      tau = TAU_SMOOTH
    else:
      tau = TAU_RELEASE
    alpha = float(dt / (tau + dt))
    self.desired_curvature = (1.0 - alpha) * self._prev_curvature + alpha * target_curvature
    if abs(target_curvature) == 0.0 and abs(self.desired_curvature) <= ACTIVE_CURVATURE_EPS:
      self.desired_curvature = 0.0
    self._prev_curvature = self.desired_curvature
    self.desired_lat_accel = float(self.desired_curvature * (v_ego * v_ego))

    # 13. Active Actuation & Solid Cluster Line Visibility States (>= 1.0s minimum yellow hold)
    self.active = bool(self.nudge_active or abs(self.desired_curvature) > ACTIVE_CURVATURE_EPS)
    if not self.active and not self.nudge_active:
      self.boundary_curvature = 0.0

    if (is_left_nudge and self.nudge_active) or (self.active and self.desired_curvature > ACTIVE_CURVATURE_EPS):
      active_side = "left"
    elif (is_right_nudge and self.nudge_active) or (self.active and self.desired_curvature < -ACTIVE_CURVATURE_EPS):
      active_side = "right"
    else:
      active_side = None

    if self._blinker_cooldown_left > 0.0:
      self._left_yellow_hold = 0.0
    elif not steer_suppressed and active_side == "left":
      self._left_yellow_hold = YELLOW_LINE_HOLD_S
      self._right_yellow_hold = 0.0
    else:
      self._left_yellow_hold = max(0.0, self._left_yellow_hold - dt)

    if self._blinker_cooldown_right > 0.0:
      self._right_yellow_hold = 0.0
    elif not steer_suppressed and active_side == "right":
      self._right_yellow_hold = YELLOW_LINE_HOLD_S
      self._left_yellow_hold = 0.0
    else:
      self._right_yellow_hold = max(0.0, self._right_yellow_hold - dt)

    left_yellow = (not steer_suppressed and active_side == "left") or (self._left_yellow_hold > 0.0)
    right_yellow = (not steer_suppressed and active_side == "right") or (self._right_yellow_hold > 0.0)

    # Suppressed during sharp cornering (>= 45 deg) for white lines and new triggers,
    # while allowing an already-latched >= 1.0s amber hold to finish cleanly without truncation.
    if self._blinker_cooldown_left > 0.0:
      self.left_line_state = 0
    elif left_yellow:
      self.left_line_state = 3  # Amber / Yellow
    elif not steer_suppressed and left_line_visible:
      self.left_line_state = 1  # Solid White
    else:
      self.left_line_state = 0

    if self._blinker_cooldown_right > 0.0:
      self.right_line_state = 0
    elif right_yellow:
      self.right_line_state = 3  # Amber / Yellow
    elif not steer_suppressed and right_line_visible:
      self.right_line_state = 1  # Solid White
    else:
      self.right_line_state = 0

    # 14. UI Gradient State
    if self.active or left_yellow or right_yellow:
      proximity = max(0.0, min(1.0, abs(self.desired_lat_accel) / max(MAX_LAT_ACCEL, 1e-3)))
      self.gradient_opacity = float(0.35 + 0.65 * proximity)
    else:
      self.gradient_opacity = 0.0


def extract_lane_keep_visual_state(lateral_state) -> tuple[bool, bool, float, float]:
  """
  Single-source UI helper reading 100 Hz LKA visual state published by controlsd
  on StarPilotLateralState.
  Returns (is_left_nudge, is_right_nudge, opacity, ribbon_width).
  """
  if lateral_state is None or not bool(getattr(lateral_state, "lkaArmed", False)):
    return False, False, 0.0, DEFAULT_GRADIENT_WIDTH

  is_left = int(getattr(lateral_state, "lkaLeftLineState", 0)) == 3
  is_right = int(getattr(lateral_state, "lkaRightLineState", 0)) == 3
  width = float(getattr(lateral_state, "lkaGradientWidth", DEFAULT_GRADIENT_WIDTH) or DEFAULT_GRADIENT_WIDTH)
  if not (is_left or is_right):
    return False, False, 0.0, width

  opacity = float(getattr(lateral_state, "lkaGradientOpacity", 0.0))
  return is_left, is_right, opacity, width
