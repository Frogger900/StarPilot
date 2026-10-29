import numpy as np
from enum import IntEnum

from openpilot.common.constants import CV
DT_CTRL = 0.01


class LaneKeepMode(IntEnum):
  OFF = 0
  ROAD_EDGE_ONLY = 1
  LANE_LINE_AND_ROAD_EDGE = 2


class LaneKeepSensitivity(IntEnum):
  LATE = 0
  NORMAL = 1
  PROACTIVE = 2


class LaneKeepAlert(IntEnum):
  NONE = 0
  KEEP_HANDS_ON_WHEEL = 1
  EMERGENCY_DEPARTURE = 2


# Default vehicle geometry for Subaru Legacy Global Gen2
DEFAULT_VEHICLE_HALF_WIDTH = 0.92  # meters (centerline to outer edge of tire)
DEFAULT_CAMERA_OFFSET = 0.04       # meters

# Dynamics and Acceleration Limits
MAX_LAT_ACCEL = 0.60               # m/s^2 gentle boundary nudge ceiling
MAX_STEER_RATE_SCALE = 1.35        # +35% authority for emergency road edge mitigation
OVERRIDE_TORQUE_THRESHOLD = 0.5    # Nm (light hand pressure yields assist)
OVERRIDE_CAN_TORQUE_THRESHOLD = 70.0  # Raw CAN torque units (softer than MADS so light hand steering yields LKA)
CORNERING_GATE_ANGLE_DEG = 45.0    # degrees (suppress assist during active turns)
APEX_CUT_BUFFER_M = 0.25           # meters (reduce trigger margin on inside curve apex)
POST_BLINKER_COOLDOWN_S = 2.0      # seconds to wait after turn signal auto-cancels

# S-Curve & 20 Hz Vision Velocity Smoothing Filters
TAU_SMOOTH = 0.15
TAU_OVERRIDE_RELEASE = 0.04
TAU_VEL = 0.15
MAX_D_DOT_MPS = 1.5
TTC_LINE_PROXIMITY_M = 0.22
TTC_EDGE_PROXIMITY_M = 0.30
TTC_CONFIRM_TIME_S = 0.08
NUDGE_RELEASE_HYSTERESIS_M = 0.04
ACTIVE_CURVATURE_EPS = 1.2e-4


class LaneKeepController:
  def __init__(self, half_width: float = DEFAULT_VEHICLE_HALF_WIDTH, camera_offset: float = DEFAULT_CAMERA_OFFSET):
    self.w_half = half_width
    self.camera_offset = camera_offset

    # State variables
    self.active = False
    self.rdm_active = False
    self.desired_curvature = 0.0
    self.desired_lat_accel = 0.0
    self.left_line_state = 0   # 0: Off, 1: White, 3: Amber
    self.right_line_state = 0  # 0: Off, 1: White, 3: Amber
    self.alert = LaneKeepAlert.NONE

    # UI Feedback properties
    self.gradient_opacity = 0.0
    self.gradient_width = 0.30
    self.is_red_gradient = False

    # Internal tracking
    self._prev_curvature = 0.0
    self._prev_d_left = None
    self._prev_d_right = None
    self._prev_d_edge_left = None
    self._prev_d_edge_right = None
    self._d_dot_left = 0.0
    self._d_dot_right = 0.0
    self._d_dot_edge_left = 0.0
    self._d_dot_edge_right = 0.0
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._ttc_edge_left_timer = 0.0
    self._ttc_edge_right_timer = 0.0
    self._nudge_side = None
    self._prev_v_ego = 0.0
    self._intervention_timer = 0.0
    self._rdm_timer = 0.0
    self._lockout = False
    self._blinker_cooldown_left = 0.0
    self._blinker_cooldown_right = 0.0
    self._speed_armed = False

  def reset(self, full: bool = False):
    self.active = False
    self.rdm_active = False
    self.desired_curvature = 0.0
    self.desired_lat_accel = 0.0
    self.left_line_state = 0
    self.right_line_state = 0
    self.alert = LaneKeepAlert.NONE
    self.gradient_opacity = 0.0
    self.is_red_gradient = False
    self._prev_curvature = 0.0
    self._prev_d_left = None
    self._prev_d_right = None
    self._prev_d_edge_left = None
    self._prev_d_edge_right = None
    self._d_dot_left = 0.0
    self._d_dot_right = 0.0
    self._d_dot_edge_left = 0.0
    self._d_dot_edge_right = 0.0
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._ttc_edge_left_timer = 0.0
    self._ttc_edge_right_timer = 0.0
    self._nudge_side = None
    self._intervention_timer = 0.0
    self._rdm_timer = 0.0
    self._lockout = False
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
    mode: int = LaneKeepMode.LANE_LINE_AND_ROAD_EDGE,
    sensitivity: int = LaneKeepSensitivity.NORMAL,
    min_activation_speed_mph: float = 25.0,
    dt: float = DT_CTRL,
    model_updated: bool = True,
  ):
    """
    Core LKA / RDM cycle executed at controls rate (100 Hz / DT_CTRL).
    Openpilot curvature sign convention:
      +curvature (> 0) = RIGHT turn (negative steeringAngleDeg)
      -curvature (< 0) = LEFT turn  (positive steeringAngleDeg)
    """
    # 1. Master enable / mode check
    if mode == LaneKeepMode.OFF:
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

    # 6. Sensitivity Trigger Thresholds
    if sensitivity == LaneKeepSensitivity.LATE:
      d_thresh = 0.00
      ttc_thresh = 0.85
      self.gradient_width = 0.15
    elif sensitivity == LaneKeepSensitivity.PROACTIVE:
      d_thresh = 0.33
      ttc_thresh = 1.40
      self.gradient_width = 0.45
    else:  # NORMAL (Default)
      d_thresh = 0.20
      ttc_thresh = 1.15
      self.gradient_width = 0.30

    d_edge_thresh = 0.28
    ttc_edge_thresh = 1.40

    # 7. Extract Vision Data
    left_prob = float(model_v2.laneLineProbs[1]) if len(model_v2.laneLineProbs) > 1 else 0.0
    right_prob = float(model_v2.laneLineProbs[2]) if len(model_v2.laneLineProbs) > 2 else 0.0
    left_line_visible = left_prob >= 0.40
    right_line_visible = right_prob >= 0.40

    left_y = float(model_v2.laneLines[1].y[0]) if len(model_v2.laneLines) > 1 and len(model_v2.laneLines[1].y) > 0 else -1.8
    right_y = float(model_v2.laneLines[2].y[0]) if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[2].y) > 0 else 1.8

    edge_left_y = float(model_v2.roadEdges[0].y[0]) if len(model_v2.roadEdges) > 0 and len(model_v2.roadEdges[0].y) > 0 else -2.5
    edge_right_y = float(model_v2.roadEdges[1].y[0]) if len(model_v2.roadEdges) > 1 and len(model_v2.roadEdges[1].y) > 0 else 2.5
    edge_left_std = float(model_v2.roadEdgeStds[0]) if len(model_v2.roadEdgeStds) > 0 else 1.0
    edge_right_std = float(model_v2.roadEdgeStds[1]) if len(model_v2.roadEdgeStds) > 1 else 1.0
    edge_left_valid = edge_left_std < 0.50
    edge_right_valid = edge_right_std < 0.50

    # Distance to tire outer edges
    d_left = -left_y - (self.w_half + self.camera_offset)
    d_right = right_y - (self.w_half - self.camera_offset)
    d_edge_left = -edge_left_y - (self.w_half + self.camera_offset)
    d_edge_right = edge_right_y - (self.w_half - self.camera_offset)

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

      if edge_left_valid:
        if self._prev_d_edge_left is not None:
          raw_el = float(np.clip((d_edge_left - self._prev_d_edge_left) / dt_m, -MAX_D_DOT_MPS, MAX_D_DOT_MPS))
          self._d_dot_edge_left = (1.0 - alpha_v) * self._d_dot_edge_left + alpha_v * raw_el
        self._prev_d_edge_left = d_edge_left
      else:
        self._prev_d_edge_left = None
        self._d_dot_edge_left = 0.0

      if edge_right_valid:
        if self._prev_d_edge_right is not None:
          raw_er = float(np.clip((d_edge_right - self._prev_d_edge_right) / dt_m, -MAX_D_DOT_MPS, MAX_D_DOT_MPS))
          self._d_dot_edge_right = (1.0 - alpha_v) * self._d_dot_edge_right + alpha_v * raw_er
        self._prev_d_edge_right = d_edge_right
      else:
        self._prev_d_edge_right = None
        self._d_dot_edge_right = 0.0

    d_dot_left = self._d_dot_left
    d_dot_right = self._d_dot_right
    d_dot_edge_left = self._d_dot_edge_left
    d_dot_edge_right = self._d_dot_edge_right

    # Time-To-Cross (TTC) for painted lines and road edges
    ttc_left = max(0.0, d_left) / abs(d_dot_left) if d_dot_left < -0.08 else 99.0
    ttc_right = max(0.0, d_right) / abs(d_dot_right) if d_dot_right < -0.08 else 99.0
    ttc_edge_left = max(0.0, d_edge_left) / abs(d_dot_edge_left) if d_dot_edge_left < -0.08 else 99.0
    ttc_edge_right = max(0.0, d_edge_right) / abs(d_dot_edge_right) if d_dot_edge_right < -0.08 else 99.0

    # Curve Apex Allowance: reduce trigger margin on inside of curve (+road_curv = right curve, -road_curv = left curve)
    road_curv = float(model_v2.action.desiredCurvature) if hasattr(model_v2, "action") else 0.0
    d_thresh_left = max(0.0, d_thresh - (APEX_CUT_BUFFER_M if road_curv < -0.001 else 0.0))
    d_thresh_right = max(0.0, d_thresh - (APEX_CUT_BUFFER_M if road_curv > 0.001 else 0.0))

    # Proximity-gated & debounced TTC triggers (prevents mid-lane and far road-edge false triggers)
    if left_line_visible and d_left < (d_thresh_left + TTC_LINE_PROXIMITY_M) and ttc_left < ttc_thresh:
      self._ttc_left_timer += dt
    else:
      self._ttc_left_timer = 0.0

    if right_line_visible and d_right < (d_thresh_right + TTC_LINE_PROXIMITY_M) and ttc_right < ttc_thresh:
      self._ttc_right_timer += dt
    else:
      self._ttc_right_timer = 0.0

    if edge_left_valid and d_edge_left < (d_edge_thresh + TTC_EDGE_PROXIMITY_M) and ttc_edge_left < ttc_edge_thresh:
      self._ttc_edge_left_timer += dt
    else:
      self._ttc_edge_left_timer = 0.0

    if edge_right_valid and d_edge_right < (d_edge_thresh + TTC_EDGE_PROXIMITY_M) and ttc_edge_right < ttc_edge_thresh:
      self._ttc_edge_right_timer += dt
    else:
      self._ttc_edge_right_timer = 0.0

    left_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "left" and d_dot_left <= 0.05) else 0.0
    right_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "right" and d_dot_right <= 0.05) else 0.0
    edge_left_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "left" and d_dot_edge_left <= 0.05) else 0.0
    edge_right_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "right" and d_dot_edge_right <= 0.05) else 0.0

    # 8. Trigger Detection (Left vs. Right)
    trigger_left_line = (
      mode == LaneKeepMode.LANE_LINE_AND_ROAD_EDGE and
      left_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_left <= 0.0 and
      (d_left < (d_thresh_left + left_hyst) or self._ttc_left_timer >= TTC_CONFIRM_TIME_S)
    )

    trigger_right_line = (
      mode == LaneKeepMode.LANE_LINE_AND_ROAD_EDGE and
      right_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_right <= 0.0 and
      (d_right < (d_thresh_right + right_hyst) or self._ttc_right_timer >= TTC_CONFIRM_TIME_S)
    )

    # Road Edge Mitigation (RDM) Fallback
    trigger_rdm_left = (
      edge_left_valid and
      not steer_suppressed and
      self._blinker_cooldown_left <= 0.0 and
      (
        d_edge_left < (d_edge_thresh + edge_left_hyst) or
        self._ttc_edge_left_timer >= TTC_CONFIRM_TIME_S or
        (left_line_visible and d_left < 0.0 and d_edge_left < 0.45)
      )
    )

    trigger_rdm_right = (
      edge_right_valid and
      not steer_suppressed and
      self._blinker_cooldown_right <= 0.0 and
      (
        d_edge_right < (d_edge_thresh + edge_right_hyst) or
        self._ttc_edge_right_timer >= TTC_CONFIRM_TIME_S or
        (right_line_visible and d_right < 0.0 and d_edge_right < 0.45)
      )
    )

    # 9. Driver Override & Lockout Evaluation
    # steering_pressed is openpilot's calibrated boolean from vehicle CAN (e.g. > 80 CAN torque units on Subaru).
    # Softer CAN threshold (70 units) ensures light intentional steering effortlessly yields LKA nudges.
    driver_override = steering_pressed or (
      abs(steering_torque) > OVERRIDE_CAN_TORQUE_THRESHOLD
      if abs(steering_torque) > 5.0
      else abs(steering_torque) > OVERRIDE_TORQUE_THRESHOLD
    )

    if self._lockout:
      left_clear = (not left_line_visible or d_left > d_thresh_left + 0.15) and (edge_left_std >= 0.50 or d_edge_left > d_edge_thresh + 0.15)
      right_clear = (not right_line_visible or d_right > d_thresh_right + 0.15) and (edge_right_std >= 0.50 or d_edge_right > d_edge_thresh + 0.15)
      # Clear lockout once driver applies torque or vehicle returns safely inside boundaries
      if driver_override or (left_clear and right_clear):
        self._lockout = False
        self._intervention_timer = 0.0
        self._rdm_timer = 0.0

    # 10. Compute Corrective Spring (Curvature)
    target_curvature = 0.0
    is_left_nudge = False
    is_right_nudge = False
    rdm_active = False

    # Gentle Proportional and Derivative spring gains (subtle boundary nudge, ~50% softer)
    kp = 0.0034
    kd = 0.0012

    if not driver_override:
      if trigger_rdm_left or trigger_left_line:
        is_left_nudge = True
        rdm_active = bool(trigger_rdm_left)
        if not self._lockout:
          d_err = (d_edge_thresh + edge_left_hyst) - d_edge_left if trigger_rdm_left else (d_thresh_left + left_hyst) - d_left
          d_err = max(0.0, d_err)
          vel_left = max(0.0, -d_dot_edge_left) if trigger_rdm_left else max(0.0, -d_dot_left)
          # Positive curvature turns RIGHT away from left edge
          target_curvature = float(kp * d_err + kd * vel_left)
      elif trigger_rdm_right or trigger_right_line:
        is_right_nudge = True
        rdm_active = bool(trigger_rdm_right)
        if not self._lockout:
          d_err = (d_edge_thresh + edge_right_hyst) - d_edge_right if trigger_rdm_right else (d_thresh_right + right_hyst) - d_right
          d_err = max(0.0, d_err)
          vel_right = max(0.0, -d_dot_edge_right) if trigger_rdm_right else max(0.0, -d_dot_right)
          # Negative curvature turns LEFT away from right edge
          target_curvature = -float(kp * d_err + kd * vel_right)

    if is_left_nudge:
      self._nudge_side = "left"
    elif is_right_nudge:
      self._nudge_side = "right"
    else:
      self._nudge_side = None

    # Clamp lateral acceleration to ceiling (1.1 m/s^2 standard, 1.43 m/s^2 RDM)
    max_accel = MAX_LAT_ACCEL * (MAX_STEER_RATE_SCALE if rdm_active else 1.0)
    max_curvature = max_accel / max(v_ego * v_ego, 1.0)
    target_curvature = float(np.clip(target_curvature, -max_curvature, max_curvature))

    # 11. Anti-Lazy & RDM Authority Timer Progression
    nudge_active = is_left_nudge or is_right_nudge
    if nudge_active and not driver_override and not self._lockout:
      self._intervention_timer += dt
      if rdm_active:
        self._rdm_timer += dt
      else:
        self._rdm_timer = 0.0
    elif not self._lockout:
      self._intervention_timer = max(0.0, self._intervention_timer - 2.0 * dt)
      self._rdm_timer = max(0.0, self._rdm_timer - 2.0 * dt)

    # Alert Stage Evaluation (18s authority ceiling for RDM, 10s for standard lane-line LKA)
    lockout_limit = 18.0 if rdm_active else 10.0
    if self._intervention_timer >= lockout_limit:
      # Stage 3: Cutoff & Lockout (prevent hands-off abuse)
      self._lockout = True
      target_curvature = 0.0
      self.alert = LaneKeepAlert.EMERGENCY_DEPARTURE
    elif self._lockout:
      target_curvature = 0.0
      self.alert = LaneKeepAlert.EMERGENCY_DEPARTURE
    elif rdm_active and self._rdm_timer >= 15.0:
      # RDM Emergency: sustained road-edge intervention >= 15s (continues steering up to 18s)
      self.alert = LaneKeepAlert.EMERGENCY_DEPARTURE
    elif self._intervention_timer >= 8.0:
      # Stage 2: Approaching timeout prompt (8s - 10s)
      self.alert = LaneKeepAlert.KEEP_HANDS_ON_WHEEL
    else:
      # Stage 1: Silent gentle nudge (0s - 8s)
      self.alert = LaneKeepAlert.NONE

    # 12. S-Curve Envelope Smoothing Filter (fast release on driver override, no sticky EPS servo tail)
    tau = TAU_OVERRIDE_RELEASE if driver_override else TAU_SMOOTH
    alpha = float(dt / (tau + dt))
    self.desired_curvature = (1.0 - alpha) * self._prev_curvature + alpha * target_curvature
    if not nudge_active and abs(self.desired_curvature) <= ACTIVE_CURVATURE_EPS:
      self.desired_curvature = 0.0
    self._prev_curvature = self.desired_curvature
    self.desired_lat_accel = float(self.desired_curvature * (v_ego * v_ego))

    # 13. Cluster Line Visibility States
    # Suppressed during sharp cornering (>= 45 deg). Amber (3) fires for both lane-line LKA
    # and unstriped road-edge RDM interventions.
    if steer_suppressed:
      self.left_line_state = 0
      self.right_line_state = 0
    else:
      if self._blinker_cooldown_left > 0.0:
        self.left_line_state = 0
      elif is_left_nudge:
        self.left_line_state = 3  # Amber / Yellow
      elif left_line_visible:
        self.left_line_state = 1  # Solid White
      else:
        self.left_line_state = 0

      if self._blinker_cooldown_right > 0.0:
        self.right_line_state = 0
      elif is_right_nudge:
        self.right_line_state = 3  # Amber / Yellow
      elif right_line_visible:
        self.right_line_state = 1  # Solid White
      else:
        self.right_line_state = 0

    # 14. UI Gradient State
    self.active = bool((nudge_active and abs(target_curvature) > 0.0) or abs(self.desired_curvature) > ACTIVE_CURVATURE_EPS)
    self.rdm_active = rdm_active
    self.is_red_gradient = rdm_active

    if self.active or nudge_active:
      proximity = max(0.0, min(1.0, abs(self.desired_lat_accel) / max(max_accel, 1e-3)))
      self.gradient_opacity = float(0.35 + 0.65 * proximity)
    else:
      self.gradient_opacity = 0.0


def extract_lane_keep_visual_state(lateral_state) -> tuple[bool, bool, bool, float, float]:
  """
  Single-source UI helper reading 100 Hz LKA visual state published by controlsd
  on StarPilotLateralState.
  Returns (is_left_nudge, is_right_nudge, is_red, opacity, ribbon_width).
  """
  if lateral_state is None or not bool(getattr(lateral_state, "lkaArmed", False)):
    return False, False, False, 0.0, 0.30

  is_left = int(getattr(lateral_state, "lkaLeftLineState", 0)) == 3
  is_right = int(getattr(lateral_state, "lkaRightLineState", 0)) == 3
  width = float(getattr(lateral_state, "lkaGradientWidth", 0.30) or 0.30)
  if not (is_left or is_right):
    return False, False, False, 0.0, width

  is_red = bool(getattr(lateral_state, "lkaRdmActive", False))
  opacity = float(getattr(lateral_state, "lkaGradientOpacity", 0.0))
  return is_left, is_right, is_red, opacity, width
