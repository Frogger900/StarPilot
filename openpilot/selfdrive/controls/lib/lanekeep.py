from collections import deque
from enum import IntEnum
import numpy as np

from openpilot.common.constants import CV

DT_CTRL = 0.01


class LaneKeepAlert(IntEnum):
  NONE = 0
  KEEP_HANDS_ON_WHEEL = 1
  EMERGENCY_DEPARTURE = 2
  KEEP_HANDS_ON_WHEEL_CHIME = 3


# Default Subaru angle-steering vehicle geometry
DEFAULT_VEHICLE_HALF_WIDTH = 0.92  # meters (centerline to outer edge of tire)
DEFAULT_CAMERA_OFFSET = 0.04       # meters

# Standard Painted-Line Trigger Thresholds (~3 in margin from painted line when both lines visible)
STANDARD_LINE_MARGIN_M = 0.07
STANDARD_TTC_PROX_M = 0.06
STANDARD_TTC_THRESH_S = 0.75
# Single-Line Trigger Thresholds (reduced sensitivity when only one painted line is visible)
SINGLE_LINE_MARGIN_M = -0.02
SINGLE_LINE_TTC_PROX_M = 0.04
ROAD_EDGE_MAX_STD = 0.45
UNMARKED_EDGE_BUFFER_M = 0.25
DEFAULT_GRADIENT_WIDTH = 0.30

# Dynamics and Acceleration Limits
MAX_LAT_ACCEL = 0.40                       # m/s^2 gentle boundary-parallel arrest ceiling
OVERRIDE_TORQUE_THRESHOLD = 0.5            # Nm (used only when torque_unit_nm=True)
OVERRIDE_OPPOSING_CAN_TORQUE = 175.0       # Raw CAN torque opposing the nudge (above passive hand drag)
OVERRIDE_ASSISTING_CAN_TORQUE = 65.0       # Delta CAN torque steering with the nudge / inward recovery
OVERRIDE_HOLD_CAN_TORQUE = 110.0           # Raw CAN torque keeping override cooldown latched
OVERRIDE_CONFIRM_TIME_S = 0.05             # 50 ms debounce for CAN torque override
OVERRIDE_COOLDOWN_S = 1.20                 # Latched cooldown after override to prevent ON-OFF-ON pulsing
MAX_CURVE_HOLD_BASE_TORQUE = 190.0         # Max pre-departure curve-holding CAN torque baseline
IDLE_TORQUE_TAU_S = 0.12                   # Smoothing time constant for pre-departure curve-holding torque
POST_ARREST_STRAIGHTEN_CAN_TORQUE = 95.0   # Deliberate opposing CAN torque releasing hold once drift stops
POST_ARREST_INWARD_CAN_TORQUE = 55.0       # Inward recovery delta CAN torque releasing hold immediately
POST_ARREST_CONFIRM_TIME_S = 0.04          # 40 ms debounce for post-arrest release
POST_ARREST_COOLDOWN_S = 0.50              # Brief cooldown after driver takes over from an arrested drift
INWARD_PAUSE_COOLDOWN_S = 0.35             # Brief anti-chatter cooldown after driver inward release
CORNERING_GATE_ANGLE_DEG = 20.0            # degrees (suppress assist at and above 20 deg of wheel turn)
POST_CORNERING_COOLDOWN_S = 1.5            # seconds to wait after unwinding below cornering angle gate
MAX_ENTRY_RECEDE_SPEED_MPS = 0.06          # do not start a fresh LKA episode if already receding faster than this
LKA_MAX_INWARD_ACCEL_MPS2 = 0.35           # m/s^2 cap on inward guidance beyond entry anchor
LKA_RECEDE_TAPER_END_MPS = 0.08            # m/s receding speed where inward guidance finishes tapering to anchor
LKA_RECEDE_RELEASE_SPEED_MPS = 0.07        # m/s sustained inward speed confirming departure is reversed
LKA_RECEDE_RELEASE_CONFIRM_S = 0.14        # seconds of sustained inward receding before crisp release
LKA_RECEDE_RELEASE_PROGRESS_M = 0.03       # meters recovered from episode min-d to reject 20 Hz vision jitter
LKA_RECEDE_RELEASE_COOLDOWN_S = 1.0        # anti-chatter hold while gliding back inside lane
LKA_RECEDE_RELEASE_MIN_COOLDOWN_S = 0.25   # minimum debounce before a fresh outward re-drift can re-trigger
LKA_RECEDE_REDRIFT_SPEED_MPS = -0.08       # outward closing speed that breaks recede-release hold early
APEX_CUT_CURVATURE_THRESH = 0.002          # 1/m (~500 m curve radius) before apex margin relaxation applies
APEX_CUT_BUFFER_M = 0.06                   # meters (modest inside-apex margin reduction without deep line cut)
POST_BLINKER_COOLDOWN_S = 2.0              # seconds to wait after turn signal auto-cancels

# Rolling 30-Second Anti-Lazy Hands-Off Timeout (5s visual / 10s bouncy chime / 15s lockout)
HANDS_OFF_WINDOW_S = 30.0
HANDS_OFF_WARN_S = 5.0
HANDS_OFF_CHIME_S = 10.0
HANDS_OFF_CHIME_GRACE_S = HANDS_OFF_CHIME_S - HANDS_OFF_WARN_S  # 5.0s grace after 5.0s warning = 10.0s chime
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
MIN_TTC_CLOSING_SPEED_MPS = 0.18
TTC_CONFIRM_TIME_S = 0.14
NUDGE_RELEASE_HYSTERESIS_M = 0.02
ACTIVE_CURVATURE_EPS = 1.0e-4
YELLOW_LINE_HOLD_S = 1.05
LANE_LINE_PROB_THRESHOLD = 0.55
LANE_LINE_PROB_OFF_THRESHOLD = 0.40
LANE_LINE_DISAPPEAR_TIME_S = 0.25
LANE_LINE_REAPPEAR_TIME_S = 1.0
INCH_TO_M = 0.0254
MAX_LINE_OFFSET_IN = 6.0


def _update_line_visibility(prev_vis: bool | None, timer: float, prob: float | bool, dt: float) -> tuple[bool, float]:
  """Apply confidence hysteresis and asymmetric debounce to lane line visibility."""
  val = (1.0 if prob else 0.0) if isinstance(prob, bool) else float(prob)
  if prev_vis is None:
    return val >= LANE_LINE_PROB_THRESHOLD, 0.0
  if prev_vis:
    if val < LANE_LINE_PROB_OFF_THRESHOLD:
      timer += max(0.0, dt)
      if timer >= LANE_LINE_DISAPPEAR_TIME_S - 1e-6:
        return False, 0.0
      return True, timer
    return True, 0.0
  if val >= LANE_LINE_PROB_THRESHOLD:
    timer += max(0.0, dt)
    if timer >= LANE_LINE_REAPPEAR_TIME_S - 1e-6:
      return True, 0.0
    return False, timer
  return False, 0.0


def _update_recede_cooldown(
  cooldown: float, elapsed: float, line_visible: bool, d: float, d_thresh: float, d_dot: float, dt: float
) -> tuple[float, float]:
  """Hold post-recede anti-chatter cooldown until safely back inside lane or actively re-drifting outward."""
  if cooldown <= 0.0:
    return 0.0, 0.0
  elapsed += dt
  if (
    not line_visible or
    d > d_thresh + 0.05 or
    (elapsed >= LKA_RECEDE_RELEASE_MIN_COOLDOWN_S and d_dot < LKA_RECEDE_REDRIFT_SPEED_MPS)
  ):
    return 0.0, 0.0
  return max(0.0, cooldown - dt), elapsed


def compute_lka_guided_curvature(
  current_curvature: float,
  anchor_curvature: float,
  lka_desired_curvature: float,
  d_dot: float,
  v_ego: float,
  nudge_right: bool,
) -> tuple[float, float]:
  """Compute strictly one-way, entry-anchored inward LKA curvature without model_v2 curve takeover or ratcheting.

  Returns (guided_curvature, updated_anchor_curvature):
  - anchor tracks outward steering downward (min in inward coords) so push stays bounded,
    and never climbs inward as LKA steers (preventing cross-lane ratcheting).
  - guided is strictly >= current_curvature in inward coordinates (never pulls outward).
  """
  sign = 1.0 if nudge_right else -1.0
  v2_eff = max(float(v_ego) * float(v_ego), 25.0)
  max_inward_push = LKA_MAX_INWARD_ACCEL_MPS2 / v2_eff

  cur_in = sign * float(current_curvature)
  anchor_in = min(sign * float(anchor_curvature), cur_in)
  taper = max(0.0, 1.0 - float(d_dot) / LKA_RECEDE_TAPER_END_MPS) if d_dot > 0.0 else 1.0
  lka_inward = max(0.0, sign * float(lka_desired_curvature))
  inward_push = min(max_inward_push, lka_inward * taper)

  cmd_in = max(cur_in, anchor_in + inward_push)
  return float(sign * cmd_in), float(sign * anchor_in)


class LaneKeepController:
  def __init__(self, half_width: float = DEFAULT_VEHICLE_HALF_WIDTH, camera_offset: float = DEFAULT_CAMERA_OFFSET):
    self.w_half = half_width
    self.camera_offset = camera_offset

    # State variables
    self.active = False
    self.nudge_active = False
    self.desired_curvature = 0.0
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
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._override_confirm_timer = 0.0
    self._straighten_confirm_timer = 0.0
    self._inward_confirm_timer = 0.0
    self._override_cooldown_left = 0.0
    self._override_cooldown_right = 0.0
    self._left_yellow_hold = 0.0
    self._right_yellow_hold = 0.0
    self._nudge_side = None
    self._episode_timer = 0.0
    self._recede_confirm_timer = 0.0
    self._episode_min_d: float | None = None
    self._recede_cooldown_left = 0.0
    self._recede_cooldown_right = 0.0
    self._recede_elapsed_left = 0.0
    self._recede_elapsed_right = 0.0
    self._window_samples: deque[tuple[float, float]] = deque()
    self._window_total_dt = 0.0
    self._window_active_dt = 0.0
    self._hands_off_warned = False
    self._hands_off_grace_timer = 0.0
    self._hands_on_confirm_timer = 0.0
    self._lockout = False
    self._blinker_cooldown_left = 0.0
    self._blinker_cooldown_right = 0.0
    self._cornering_cooldown = 0.0
    self._speed_armed = False
    self._left_line_vis: bool | None = None
    self._right_line_vis: bool | None = None
    self._left_line_on_timer = 0.0
    self._right_line_on_timer = 0.0
    self._idle_torque = 0.0
    self._prev_candidate_side: str | None = None
    self.assist_torque_baseline = 0.0

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
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._override_confirm_timer = 0.0
    self._straighten_confirm_timer = 0.0
    self._inward_confirm_timer = 0.0
    self._override_cooldown_left = 0.0
    self._override_cooldown_right = 0.0
    self._left_yellow_hold = 0.0
    self._right_yellow_hold = 0.0
    self._nudge_side = None
    self._episode_timer = 0.0
    self._recede_confirm_timer = 0.0
    self._episode_min_d: float | None = None
    self._recede_cooldown_left = 0.0
    self._recede_cooldown_right = 0.0
    self._recede_elapsed_left = 0.0
    self._recede_elapsed_right = 0.0
    self._clear_hands_off_tracking()
    if full:
      self._speed_armed = False
      self._blinker_cooldown_left = 0.0
      self._blinker_cooldown_right = 0.0
      self._cornering_cooldown = 0.0
      self._left_line_vis = None
      self._right_line_vis = None
      self._left_line_on_timer = 0.0
      self._right_line_on_timer = 0.0
    self._idle_torque = 0.0
    self._prev_candidate_side = None
    self.assist_torque_baseline = 0.0

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
    line_offset_in: float = 0.0,
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

    # 4. Turn Signal Handlers & Cooldown Timers (either turn signal stops LKA completely on both sides)
    any_blinker = bool(left_blinker or right_blinker)
    if any_blinker:
      self._blinker_cooldown_left = POST_BLINKER_COOLDOWN_S
      self._blinker_cooldown_right = POST_BLINKER_COOLDOWN_S
    else:
      if self._blinker_cooldown_left > 0.0:
        self._blinker_cooldown_left = max(0.0, self._blinker_cooldown_left - dt)
      if self._blinker_cooldown_right > 0.0:
        self._blinker_cooldown_right = max(0.0, self._blinker_cooldown_right - dt)
    blinker_suppressed = any_blinker or (self._blinker_cooldown_left > 0.0) or (self._blinker_cooldown_right > 0.0)

    # 5. Cornering Suppression Gate (Physical wheel turned >= 20 deg + 1.5s unwind cooldown)
    if abs(steering_angle_deg) >= CORNERING_GATE_ANGLE_DEG:
      self._cornering_cooldown = POST_CORNERING_COOLDOWN_S
    elif self._cornering_cooldown > 0.0:
      self._cornering_cooldown = max(0.0, self._cornering_cooldown - dt)
    steer_suppressed = (abs(steering_angle_deg) >= CORNERING_GATE_ANGLE_DEG) or (self._cornering_cooldown > 0.0)

    # 6. Extract Painted Lane-Line & Road-Edge Vision Data
    left_prob = float(model_v2.laneLineProbs[1]) if len(model_v2.laneLineProbs) > 1 else 0.0
    right_prob = float(model_v2.laneLineProbs[2]) if len(model_v2.laneLineProbs) > 2 else 0.0
    self._left_line_vis, self._left_line_on_timer = _update_line_visibility(
      self._left_line_vis, self._left_line_on_timer, left_prob, dt
    )
    self._right_line_vis, self._right_line_on_timer = _update_line_visibility(
      self._right_line_vis, self._right_line_on_timer, right_prob, dt
    )
    left_line_visible = bool(self._left_line_vis)
    right_line_visible = bool(self._right_line_vis)

    # 7. Painted-Line Thresholds (standard when both lines visible; reduced sensitivity on single-line roads)
    both_lines_visible = left_line_visible and right_line_visible
    offset_m = float(np.clip(line_offset_in, -MAX_LINE_OFFSET_IN, MAX_LINE_OFFSET_IN)) * INCH_TO_M
    d_thresh = (STANDARD_LINE_MARGIN_M if both_lines_visible else SINGLE_LINE_MARGIN_M) + offset_m
    ttc_line_prox = STANDARD_TTC_PROX_M if both_lines_visible else SINGLE_LINE_TTC_PROX_M
    ttc_thresh = STANDARD_TTC_THRESH_S
    self.gradient_width = DEFAULT_GRADIENT_WIDTH

    left_y = float(model_v2.laneLines[1].y[0]) if len(model_v2.laneLines) > 1 and len(model_v2.laneLines[1].y) > 0 else -1.8
    right_y = float(model_v2.laneLines[2].y[0]) if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[2].y) > 0 else 1.8

    road_edges = getattr(model_v2, "roadEdges", ())
    road_edge_stds = getattr(model_v2, "roadEdgeStds", ())
    left_edge_visible = (
      len(road_edge_stds) > 0 and
      len(road_edges) > 0 and
      len(road_edges[0].y) > 0 and
      float(road_edge_stds[0]) <= ROAD_EDGE_MAX_STD
    )
    right_edge_visible = (
      len(road_edge_stds) > 1 and
      len(road_edges) > 1 and
      len(road_edges[1].y) > 0 and
      float(road_edge_stds[1]) <= ROAD_EDGE_MAX_STD
    )
    left_edge_y = float(road_edges[0].y[0]) if left_edge_visible else -1.8
    right_edge_y = float(road_edges[1].y[0]) if right_edge_visible else 1.8

    # Distance from tire outer edges to painted lane lines (and buffered physical road edges)
    d_left = -left_y - (self.w_half + self.camera_offset)
    d_right = right_y - (self.w_half - self.camera_offset)
    d_left_edge = -left_edge_y - (self.w_half + self.camera_offset) - UNMARKED_EDGE_BUFFER_M
    d_right_edge = right_edge_y - (self.w_half - self.camera_offset) - UNMARKED_EDGE_BUFFER_M

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

    if blinker_suppressed:
      self._d_dot_left = 0.0
      self._d_dot_right = 0.0
      self._prev_d_left = d_left if left_line_visible else None
      self._prev_d_right = d_right if right_line_visible else None

    d_dot_left = self._d_dot_left
    d_dot_right = self._d_dot_right

    # Time-To-Cross (TTC) for painted lines
    ttc_left = max(0.0, d_left) / abs(d_dot_left) if d_dot_left < -MIN_TTC_CLOSING_SPEED_MPS else 99.0
    ttc_right = max(0.0, d_right) / abs(d_dot_right) if d_dot_right < -MIN_TTC_CLOSING_SPEED_MPS else 99.0

    # Curve Apex Allowance: reduce trigger margin on inside of curve (+road_curv = right curve, -road_curv = left curve)
    road_curv = float(model_v2.action.desiredCurvature) if hasattr(model_v2, "action") else 0.0
    d_thresh_left = max(-0.06 + offset_m, d_thresh - APEX_CUT_BUFFER_M) if road_curv < -APEX_CUT_CURVATURE_THRESH else d_thresh
    d_thresh_right = max(-0.06 + offset_m, d_thresh - APEX_CUT_BUFFER_M) if road_curv > APEX_CUT_CURVATURE_THRESH else d_thresh
    ttc_prox_left = ttc_line_prox
    ttc_prox_right = ttc_line_prox
    if left_line_visible and right_line_visible:
      # Narrow-lane pinch guard: when lane width < ~2.45m, never trigger near lane center or ping-pong across a tight pinch
      half_clearance = 0.5 * (d_left + d_right)
      d_thresh_left = min(d_thresh_left, max(-0.04, half_clearance - 0.12))
      d_thresh_right = min(d_thresh_right, max(-0.04, half_clearance - 0.12))
      ttc_prox_left = min(ttc_line_prox, max(0.0, (half_clearance - 0.06) - d_thresh_left))
      ttc_prox_right = min(ttc_line_prox, max(0.0, (half_clearance - 0.06) - d_thresh_right))
    elif left_line_visible and right_edge_visible:
      # Single-line narrow-road pinch guard using right road edge when right painted line is absent
      half_clearance = 0.5 * (d_left + d_right_edge)
      d_thresh_left = min(d_thresh_left, max(-0.06, half_clearance - 0.12))
      ttc_prox_left = min(ttc_line_prox, max(0.0, (half_clearance - 0.06) - d_thresh_left))
    elif right_line_visible and left_edge_visible:
      # Single-line narrow-road pinch guard using left road edge when left painted line is absent
      half_clearance = 0.5 * (d_left_edge + d_right)
      d_thresh_right = min(d_thresh_right, max(-0.06, half_clearance - 0.12))
      ttc_prox_right = min(ttc_line_prox, max(0.0, (half_clearance - 0.06) - d_thresh_right))

    # Proximity-gated & debounced TTC triggers
    if left_line_visible and not steer_suppressed and not blinker_suppressed and d_left < (d_thresh_left + ttc_prox_left) and ttc_left < ttc_thresh:
      self._ttc_left_timer += dt
    else:
      self._ttc_left_timer = 0.0

    if right_line_visible and not steer_suppressed and not blinker_suppressed and d_right < (d_thresh_right + ttc_prox_right) and ttc_right < ttc_thresh:
      self._ttc_right_timer += dt
    else:
      self._ttc_right_timer = 0.0

    left_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "left" and d_dot_left < 0.03) else 0.0
    right_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "right" and d_dot_right < 0.03) else 0.0
    sustain_margin_left = min(0.055, ttc_prox_left)
    sustain_margin_right = min(0.055, ttc_prox_right)

    sustain_left_line = (
      (self._nudge_side == "left") and
      ((d_left < d_thresh_left + ttc_prox_left and d_dot_left < -0.015) or
       (d_left < d_thresh_left + sustain_margin_left and d_dot_left < 0.015))
    )
    sustain_right_line = (
      (self._nudge_side == "right") and
      ((d_right < d_thresh_right + ttc_prox_right and d_dot_right < -0.015) or
       (d_right < d_thresh_right + sustain_margin_right and d_dot_right < 0.015))
    )

    # 8. Candidate Trigger Detection (Left vs. Right painted lane lines before override cooldown gating)
    # Require the vehicle not to be already receding strongly back into the lane on initial entry.
    left_entry_allowed = (self._nudge_side == "left") or (d_dot_left < MAX_ENTRY_RECEDE_SPEED_MPS)
    right_entry_allowed = (self._nudge_side == "right") or (d_dot_right < MAX_ENTRY_RECEDE_SPEED_MPS)
    raw_trigger_left_line = (
      left_line_visible and
      not steer_suppressed and
      not blinker_suppressed and
      left_entry_allowed and
      (d_left < (d_thresh_left + left_hyst) or self._ttc_left_timer >= TTC_CONFIRM_TIME_S or sustain_left_line)
    )

    raw_trigger_right_line = (
      right_line_visible and
      not steer_suppressed and
      not blinker_suppressed and
      right_entry_allowed and
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

    self._recede_cooldown_left, self._recede_elapsed_left = _update_recede_cooldown(
      self._recede_cooldown_left, self._recede_elapsed_left, left_line_visible, d_left, d_thresh_left, d_dot_left, dt
    )
    self._recede_cooldown_right, self._recede_elapsed_right = _update_recede_cooldown(
      self._recede_cooldown_right, self._recede_elapsed_right, right_line_visible, d_right, d_thresh_right, d_dot_right, dt
    )

    if raw_trigger_left_line or self._nudge_side == "left" or self.desired_curvature > ACTIVE_CURVATURE_EPS:
      candidate_side = "left"
    elif raw_trigger_right_line or self._nudge_side == "right" or self.desired_curvature < -ACTIVE_CURVATURE_EPS:
      candidate_side = "right"
    else:
      candidate_side = None

    if candidate_side is None:
      self.assist_torque_baseline = 0.0
      if not self.active and self._override_cooldown_left <= 0.0 and self._override_cooldown_right <= 0.0:
        alpha_torque = float(dt / (IDLE_TORQUE_TAU_S + dt))
        self._idle_torque = (1.0 - alpha_torque) * self._idle_torque + alpha_torque * steering_torque
    elif self._prev_candidate_side is None and not self.active:
      if candidate_side == "left":
        self.assist_torque_baseline = float(np.clip(self._idle_torque, -MAX_CURVE_HOLD_BASE_TORQUE, 0.0))
      else:
        self.assist_torque_baseline = float(np.clip(self._idle_torque, 0.0, MAX_CURVE_HOLD_BASE_TORQUE))
    elif self._prev_candidate_side != candidate_side:
      self.assist_torque_baseline = 0.0
    self._prev_candidate_side = candidate_side

    post_arrest_release = False
    inward_pause = False
    if torque_unit_nm:
      self._override_confirm_timer = 0.0
      self._straighten_confirm_timer = 0.0
      self._inward_confirm_timer = 0.0
      driver_override = steering_pressed or abs(steering_torque) > OVERRIDE_TORQUE_THRESHOLD
    else:
      # Positive CAN steeringTorque = driver pushing LEFT; negative = driver pushing RIGHT.
      # Assisting / inward release compares against any curve-holding baseline latched before
      # departure so steady understeer torque into a turn doesn't block LKA, while straights
      # still yield at 55-65 CAN units and opposing pushes still override at 175 CAN units.
      assist_base = self.assist_torque_baseline
      if candidate_side == "left":
        torque_exceeded = steering_torque > OVERRIDE_OPPOSING_CAN_TORQUE
      elif candidate_side == "right":
        torque_exceeded = steering_torque < -OVERRIDE_OPPOSING_CAN_TORQUE
      else:
        torque_exceeded = steering_pressed or abs(steering_torque) > OVERRIDE_OPPOSING_CAN_TORQUE

      self._override_confirm_timer = (self._override_confirm_timer + dt) if torque_exceeded else 0.0
      if candidate_side is not None:
        driver_override = self._override_confirm_timer >= OVERRIDE_CONFIRM_TIME_S
      else:
        driver_override = torque_exceeded

      # Inward recovery steering pauses assist with a brief anti-chatter cooldown so releasing resumes cleanly;
      # light opposing straightening pressure releases with brief cooldown once outward drift has been arrested.
      if not driver_override and candidate_side is not None:
        left_inward = (candidate_side == "left" and steering_torque < min(0.0, assist_base) - POST_ARREST_INWARD_CAN_TORQUE)
        right_inward = (candidate_side == "right" and steering_torque > max(0.0, assist_base) + POST_ARREST_INWARD_CAN_TORQUE)
        self._inward_confirm_timer = (self._inward_confirm_timer + dt) if (left_inward or right_inward) else 0.0
        inward_pause = self._inward_confirm_timer >= POST_ARREST_CONFIRM_TIME_S
        if not inward_pause and self.active:
          left_straighten = (
            self._episode_timer >= 0.15 and
            candidate_side == "left" and
            (d_dot_left >= 0.08 or (d_left >= d_thresh_left - 0.02 and d_dot_left >= -0.005)) and
            steering_torque > POST_ARREST_STRAIGHTEN_CAN_TORQUE
          )
          right_straighten = (
            self._episode_timer >= 0.15 and
            candidate_side == "right" and
            (d_dot_right >= 0.08 or (d_right >= d_thresh_right - 0.02 and d_dot_right >= -0.005)) and
            steering_torque < -POST_ARREST_STRAIGHTEN_CAN_TORQUE
          )
          self._straighten_confirm_timer = (self._straighten_confirm_timer + dt) if (left_straighten or right_straighten) else 0.0
          if self._straighten_confirm_timer >= POST_ARREST_CONFIRM_TIME_S:
            post_arrest_release = True
            self._straighten_confirm_timer = 0.0
        else:
          self._straighten_confirm_timer = 0.0
      else:
        self._inward_confirm_timer = 0.0
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
    elif inward_pause:
      if candidate_side == "left":
        self._override_cooldown_left = max(self._override_cooldown_left, INWARD_PAUSE_COOLDOWN_S)
      elif candidate_side == "right":
        self._override_cooldown_right = max(self._override_cooldown_right, INWARD_PAUSE_COOLDOWN_S)

    # Confirm actual driver hands-on steering torque (distinct from passive wheel inertia while EPS moves,
    # and never satisfied merely by coasting back through lane center hands-off).
    if (driver_override and candidate_side is not None) or post_arrest_release:
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

    if driver_hands_on:
      self._clear_hands_off_tracking()

    trigger_left_line = raw_trigger_left_line and (self._override_cooldown_left <= 0.0) and (self._recede_cooldown_left <= 0.0)
    trigger_right_line = raw_trigger_right_line and (self._override_cooldown_right <= 0.0) and (self._recede_cooldown_right <= 0.0)

    # 10. Compute Drift-Stopping Boundary Guidance (Curvature)
    target_curvature = 0.0
    is_left_nudge = False
    is_right_nudge = False

    # Gentle drift-arresting gains: damps outward drift toward the line and fades out
    # as soon as outward drift stops (d_dot > 0) so the car parallels the line instead
    # of pushing back across the lane.
    kp = 0.0024
    kd = 0.0014

    recede_release = False
    active_nudge_side = None
    if not driver_override and not post_arrest_release and not inward_pause and not self._lockout:
      if trigger_left_line:
        active_nudge_side = "left"
      elif trigger_right_line:
        active_nudge_side = "right"

    if active_nudge_side is not None and self._nudge_side == active_nudge_side:
      cur_d = d_left if active_nudge_side == "left" else d_right
      cur_d_dot = d_dot_left if active_nudge_side == "left" else d_dot_right
      cur_d_thresh = d_thresh_left if active_nudge_side == "left" else d_thresh_right
      self._episode_min_d = cur_d if self._episode_min_d is None else min(self._episode_min_d, cur_d)
      if cur_d_dot >= LKA_RECEDE_RELEASE_SPEED_MPS:
        self._recede_confirm_timer += dt
      elif cur_d_dot < 0.0:
        self._recede_confirm_timer = 0.0
      else:
        self._recede_confirm_timer = max(0.0, self._recede_confirm_timer - dt)

      if (
        self._recede_confirm_timer >= LKA_RECEDE_RELEASE_CONFIRM_S - 1e-6 and
        (cur_d - self._episode_min_d) >= LKA_RECEDE_RELEASE_PROGRESS_M
      ) or (
        self._episode_timer >= 0.15 and
        cur_d_dot >= 0.05 and
        cur_d >= cur_d_thresh - 0.03 and
        abs(self._prev_curvature) <= ACTIVE_CURVATURE_EPS
      ):
        recede_release = True
        if active_nudge_side == "left":
          self._recede_cooldown_left = LKA_RECEDE_RELEASE_COOLDOWN_S
          self._recede_elapsed_left = 0.0
        else:
          self._recede_cooldown_right = LKA_RECEDE_RELEASE_COOLDOWN_S
          self._recede_elapsed_right = 0.0
        self._recede_confirm_timer = 0.0
        self._episode_min_d = None
    elif active_nudge_side is not None:
      self._episode_min_d = d_left if active_nudge_side == "left" else d_right
      self._recede_confirm_timer = 0.0
    else:
      self._episode_min_d = None
      self._recede_confirm_timer = 0.0

    if not driver_override and not post_arrest_release and not inward_pause and not recede_release:
      if trigger_left_line:
        is_left_nudge = True
        if not self._lockout:
          d_lim = d_thresh_left + left_hyst
          sustain_err = min(0.02, (d_thresh_left + sustain_margin_left) - d_left) if sustain_left_line else 0.0
          d_err = max(0.0, d_lim - d_left, sustain_err)
          vel_left = max(0.0, -d_dot_left)
          recede_floor = min(0.15, max(0.0, (d_thresh_left - d_left) / 0.10) * 0.15)
          recede_scale = max(recede_floor, 1.0 - (d_dot_left / 0.10)) if d_dot_left > 0.0 else 1.0
          # Positive curvature turns RIGHT away from left lane line
          target_curvature = float((kp * d_err + kd * vel_left) * recede_scale)
      elif trigger_right_line:
        is_right_nudge = True
        if not self._lockout:
          d_lim = d_thresh_right + right_hyst
          sustain_err = min(0.02, (d_thresh_right + sustain_margin_right) - d_right) if sustain_right_line else 0.0
          d_err = max(0.0, d_lim - d_right, sustain_err)
          vel_right = max(0.0, -d_dot_right)
          recede_floor = min(0.15, max(0.0, (d_thresh_right - d_right) / 0.10) * 0.15)
          recede_scale = max(recede_floor, 1.0 - (d_dot_right / 0.10)) if d_dot_right > 0.0 else 1.0
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
    override_active = driver_override or post_arrest_release or inward_pause or (self._override_cooldown_left > 0.0) or (self._override_cooldown_right > 0.0)
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
    elif self._hands_off_warned and (
      self._window_active_dt >= HANDS_OFF_CHIME_S - 1e-6 or
      self._hands_off_grace_timer >= HANDS_OFF_CHIME_GRACE_S - 1e-6
    ):
      self.alert = LaneKeepAlert.KEEP_HANDS_ON_WHEEL_CHIME
    elif self._hands_off_warned:
      self.alert = LaneKeepAlert.KEEP_HANDS_ON_WHEEL
    else:
      self.alert = LaneKeepAlert.NONE

    if blinker_suppressed:
      self.nudge_active = False
      self.active = False
      self.desired_curvature = 0.0
      self._prev_curvature = 0.0
      self.desired_lat_accel = 0.0
      self._nudge_side = None
      self._episode_timer = 0.0
      self._recede_confirm_timer = 0.0
      self._episode_min_d = None
      self._left_yellow_hold = 0.0
      self._right_yellow_hold = 0.0
      self.left_line_state = 0
      self.right_line_state = 0
      self.gradient_opacity = 0.0
      return

    self.nudge_active = bool(actively_nudging and not override_active and not self._lockout)

    # 12. Asymmetric S-Curve Envelope Smoothing Filter (gentle onset, crisp release, immediate yield on override or confirmed recede)
    if override_active or recede_release:
      self.desired_curvature = 0.0
    else:
      if abs(target_curvature) > abs(self._prev_curvature) and (target_curvature * self._prev_curvature >= 0.0):
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

    if (is_left_nudge and self.nudge_active) or (self.active and self.desired_curvature > ACTIVE_CURVATURE_EPS):
      active_side = "left"
    elif (is_right_nudge and self.nudge_active) or (self.active and self.desired_curvature < -ACTIVE_CURVATURE_EPS):
      active_side = "right"
    else:
      active_side = None

    if not steer_suppressed and active_side == "left":
      self._left_yellow_hold = YELLOW_LINE_HOLD_S
      self._right_yellow_hold = 0.0
    else:
      self._left_yellow_hold = max(0.0, self._left_yellow_hold - dt)

    if not steer_suppressed and active_side == "right":
      self._right_yellow_hold = YELLOW_LINE_HOLD_S
      self._left_yellow_hold = 0.0
    else:
      self._right_yellow_hold = max(0.0, self._right_yellow_hold - dt)

    left_yellow = (not steer_suppressed and active_side == "left") or (self._left_yellow_hold > 0.0)
    right_yellow = (not steer_suppressed and active_side == "right") or (self._right_yellow_hold > 0.0)

    # Suppressed during sharp cornering (>= 20 deg + 1.5s unwind cooldown) for white lines and new triggers,
    # while allowing an already-latched >= 1.0s amber hold to finish cleanly without truncation.
    if left_yellow:
      self.left_line_state = 3  # Amber / Yellow
    elif not steer_suppressed and left_line_visible:
      self.left_line_state = 1  # Solid White
    else:
      self.left_line_state = 0

    if right_yellow:
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
