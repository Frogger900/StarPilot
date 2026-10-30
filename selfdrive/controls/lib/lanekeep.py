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
MAX_LAT_ACCEL = 0.40                   # m/s^2 gentle boundary-parallel arrest ceiling
MAX_STEER_RATE_SCALE = 1.25            # +25% authority for emergency road edge mitigation
OVERRIDE_TORQUE_THRESHOLD = 0.5        # Nm (used only when torque_unit_nm=True)
OVERRIDE_OPPOSING_CAN_TORQUE = 140.0   # Raw CAN torque opposing the nudge (above passive hand drag)
OVERRIDE_ASSISTING_CAN_TORQUE = 185.0  # Raw CAN torque steering with the nudge / holding into curve
OVERRIDE_HOLD_CAN_TORQUE = 90.0        # Raw CAN torque keeping override cooldown latched
OVERRIDE_CONFIRM_TIME_S = 0.05         # 50 ms debounce for CAN torque override
OVERRIDE_COOLDOWN_S = 1.20             # Latched cooldown after override to prevent ON-OFF-ON pulsing
POST_ARREST_STRAIGHTEN_CAN_TORQUE = 60.0  # Gentle opposing CAN torque releasing hold once drift stops
POST_ARREST_CONFIRM_TIME_S = 0.04      # 40 ms debounce for post-arrest straightening release
POST_ARREST_COOLDOWN_S = 0.50          # Brief cooldown after driver straightens out of an arrested drift
CORNERING_GATE_ANGLE_DEG = 45.0        # degrees (suppress assist during active turns)
APEX_CUT_BUFFER_M = 0.12               # meters (reduce trigger margin on inside curve apex)
POST_BLINKER_COOLDOWN_S = 2.0          # seconds to wait after turn signal auto-cancels

# S-Curve & 20 Hz Vision Velocity Smoothing Filters
TAU_SMOOTH = 0.24
TAU_RELEASE = 0.08
TAU_RDM_RELEASE = 0.18
TAU_OVERRIDE_RELEASE = 0.04
TAU_VEL = 0.16
TAU_VEL_EDGE = 0.22
TAU_POS_EDGE = 0.10
MAX_D_DOT_MPS = 1.5
MAX_EDGE_D_DOT_MPS = 1.0
MIN_TTC_CLOSING_SPEED_MPS = 0.12
TTC_EDGE_PROXIMITY_M = 0.24
TTC_CONFIRM_TIME_S = 0.08
NUDGE_RELEASE_HYSTERESIS_M = 0.02
EDGE_RELEASE_HYSTERESIS_M = 0.09
STRIPED_ROAD_MEMORY_S = 1.80
EDGE_STD_ENTER = 0.55
EDGE_STD_EXIT = 0.68
ACTIVE_CURVATURE_EPS = 1.0e-4
YELLOW_LINE_HOLD_S = 0.40


def _estimate_boundary_curvature(line) -> float:
  """Estimate spatial road curvature (1/m, +right / -left) from a modelV2 laneLine or roadEdge."""
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
    self.rdm_active = False
    self.desired_curvature = 0.0
    self.boundary_curvature = 0.0
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
    self._filt_d_edge_left = None
    self._filt_d_edge_right = None
    self._d_dot_left = 0.0
    self._d_dot_right = 0.0
    self._d_dot_edge_left = 0.0
    self._d_dot_edge_right = 0.0
    self._edge_left_valid = False
    self._edge_right_valid = False
    self._striped_road_memory = 0.0
    self._left_line_curv = 0.0
    self._right_line_curv = 0.0
    self._left_edge_curv = 0.0
    self._right_edge_curv = 0.0
    self._curv_initialized = False
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._ttc_edge_left_timer = 0.0
    self._ttc_edge_right_timer = 0.0
    self._override_confirm_timer = 0.0
    self._straighten_confirm_timer = 0.0
    self._override_cooldown_left = 0.0
    self._override_cooldown_right = 0.0
    self._left_yellow_hold = 0.0
    self._right_yellow_hold = 0.0
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
    self.nudge_active = False
    self.rdm_active = False
    self.desired_curvature = 0.0
    self.boundary_curvature = 0.0
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
    self._filt_d_edge_left = None
    self._filt_d_edge_right = None
    self._d_dot_left = 0.0
    self._d_dot_right = 0.0
    self._d_dot_edge_left = 0.0
    self._d_dot_edge_right = 0.0
    self._edge_left_valid = False
    self._edge_right_valid = False
    self._striped_road_memory = 0.0
    self._left_line_curv = 0.0
    self._right_line_curv = 0.0
    self._left_edge_curv = 0.0
    self._right_edge_curv = 0.0
    self._curv_initialized = False
    self._dt_since_model = 0.0
    self._ttc_left_timer = 0.0
    self._ttc_right_timer = 0.0
    self._ttc_edge_left_timer = 0.0
    self._ttc_edge_right_timer = 0.0
    self._override_confirm_timer = 0.0
    self._straighten_confirm_timer = 0.0
    self._override_cooldown_left = 0.0
    self._override_cooldown_right = 0.0
    self._left_yellow_hold = 0.0
    self._right_yellow_hold = 0.0
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
    torque_unit_nm: bool = False,
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

    # 6. Sensitivity Trigger Thresholds (calibrated for narrow 2.6-2.9 m two-lane roads as well as highways)
    if sensitivity == LaneKeepSensitivity.LATE:
      d_thresh = -0.03
      ttc_line_prox = 0.08
      ttc_thresh = 0.55
      self.gradient_width = 0.15
    elif sensitivity == LaneKeepSensitivity.PROACTIVE:
      d_thresh = 0.18
      ttc_line_prox = 0.14
      ttc_thresh = 1.05
      self.gradient_width = 0.45
    else:  # NORMAL (Default: ~3 in from painted line, tight ~4 in drift lookahead)
      d_thresh = 0.07
      ttc_line_prox = 0.10
      ttc_thresh = 0.75
      self.gradient_width = 0.30

    # 7. Extract Vision Data
    left_prob = float(model_v2.laneLineProbs[1]) if len(model_v2.laneLineProbs) > 1 else 0.0
    right_prob = float(model_v2.laneLineProbs[2]) if len(model_v2.laneLineProbs) > 2 else 0.0
    left_line_visible = left_prob >= 0.40
    right_line_visible = right_prob >= 0.40

    # Painted-line priority with brief intersection paint-gap memory:
    # - On striped roads (recent confident lane markings within 1.8s), RDM stays strictly conservative
    #   so it NEVER interferes with normal painted-line LKA or triggers across brief intersection gaps.
    # - On genuinely unstriped roads / driveways (_striped_road_memory expired, opposite line < 0.25, and
    #   weak single-sided edge contrast < 0.65, or explicit ROAD_EDGE_ONLY mode), proactive RDM uses wider
    #   margin and grass-edge hysteresis for early, continuous road-edge protection.
    if (left_prob >= 0.65 and right_prob >= 0.45) or left_prob >= 0.80 or right_prob >= 0.80:
      self._striped_road_memory = STRIPED_ROAD_MEMORY_S
    elif self._striped_road_memory > 0.0:
      self._striped_road_memory = max(0.0, self._striped_road_memory - dt)

    unstriped_env_left = (mode == LaneKeepMode.ROAD_EDGE_ONLY) or (
      self._striped_road_memory <= 0.0 and right_prob < 0.25 and left_prob < 0.65
    )
    unstriped_env_right = (mode == LaneKeepMode.ROAD_EDGE_ONLY) or (
      self._striped_road_memory <= 0.0 and left_prob < 0.25 and right_prob < 0.65
    )

    d_edge_thresh_left = 0.28 if unstriped_env_left else 0.14
    d_edge_thresh_right = 0.28 if unstriped_env_right else 0.14
    ttc_edge_thresh_left = 1.15 if unstriped_env_left else 0.85
    ttc_edge_thresh_right = 1.15 if unstriped_env_right else 0.85
    ttc_edge_prox_left = TTC_EDGE_PROXIMITY_M if unstriped_env_left else 0.16
    ttc_edge_prox_right = TTC_EDGE_PROXIMITY_M if unstriped_env_right else 0.16

    left_y = float(model_v2.laneLines[1].y[0]) if len(model_v2.laneLines) > 1 and len(model_v2.laneLines[1].y) > 0 else -1.8
    right_y = float(model_v2.laneLines[2].y[0]) if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[2].y) > 0 else 1.8

    edge_left_y = float(model_v2.roadEdges[0].y[0]) if len(model_v2.roadEdges) > 0 and len(model_v2.roadEdges[0].y) > 0 else -2.5
    edge_right_y = float(model_v2.roadEdges[1].y[0]) if len(model_v2.roadEdges) > 1 and len(model_v2.roadEdges[1].y) > 0 else 2.5
    edge_left_std = float(model_v2.roadEdgeStds[0]) if len(model_v2.roadEdgeStds) > 0 else 1.0
    edge_right_std = float(model_v2.roadEdgeStds[1]) if len(model_v2.roadEdgeStds) > 1 else 1.0
    std_enter_l = EDGE_STD_ENTER if unstriped_env_left else 0.50
    std_exit_l = EDGE_STD_EXIT if unstriped_env_left else 0.50
    std_enter_r = EDGE_STD_ENTER if unstriped_env_right else 0.50
    std_exit_r = EDGE_STD_EXIT if unstriped_env_right else 0.50
    if self._edge_left_valid:
      self._edge_left_valid = edge_left_std < std_exit_l
    else:
      self._edge_left_valid = edge_left_std < std_enter_l
    if self._edge_right_valid:
      self._edge_right_valid = edge_right_std < std_exit_r
    else:
      self._edge_right_valid = edge_right_std < std_enter_r
    edge_left_valid = self._edge_left_valid
    edge_right_valid = self._edge_right_valid

    if model_updated or not self._curv_initialized:
      self._curv_initialized = True
      self._left_line_curv = _estimate_boundary_curvature(model_v2.laneLines[1]) if (left_line_visible and len(model_v2.laneLines) > 1) else 0.0
      self._right_line_curv = _estimate_boundary_curvature(model_v2.laneLines[2]) if (right_line_visible and len(model_v2.laneLines) > 2) else 0.0
      self._left_edge_curv = _estimate_boundary_curvature(model_v2.roadEdges[0]) if (edge_left_valid and len(model_v2.roadEdges) > 0) else 0.0
      self._right_edge_curv = _estimate_boundary_curvature(model_v2.roadEdges[1]) if (edge_right_valid and len(model_v2.roadEdges) > 1) else 0.0

    # Distance to tire outer edges (road-edge position lightly EMA-filtered at 20 Hz to reject grass-edge jitter,
    # while road-edge velocity is differentiated from raw_d_edge_* to preserve immediate TTC responsiveness)
    d_left = -left_y - (self.w_half + self.camera_offset)
    d_right = right_y - (self.w_half - self.camera_offset)
    raw_d_edge_left = -edge_left_y - (self.w_half + self.camera_offset)
    raw_d_edge_right = edge_right_y - (self.w_half - self.camera_offset)

    # Lateral velocities updated strictly on 20 Hz modelV2 frames with EMA filtering
    # (prevents 5x derivative spikes from differentiating 20 Hz vision at 100 Hz DT_CTRL)
    self._dt_since_model += max(0.0, dt)
    if model_updated and self._dt_since_model > 0.0:
      dt_m = max(self._dt_since_model, 1e-3)
      self._dt_since_model = 0.0
      alpha_v = float(dt_m / (TAU_VEL + dt_m))
      alpha_ve = float(dt_m / (TAU_VEL_EDGE + dt_m))
      alpha_pe = float(dt_m / (TAU_POS_EDGE + dt_m))

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
          raw_el = float(np.clip((raw_d_edge_left - self._prev_d_edge_left) / dt_m, -MAX_EDGE_D_DOT_MPS, 0.30))
          self._d_dot_edge_left = (1.0 - alpha_ve) * self._d_dot_edge_left + alpha_ve * raw_el
        self._prev_d_edge_left = raw_d_edge_left
        if self._filt_d_edge_left is None:
          self._filt_d_edge_left = raw_d_edge_left
        else:
          self._filt_d_edge_left = (1.0 - alpha_pe) * self._filt_d_edge_left + alpha_pe * raw_d_edge_left
      else:
        self._prev_d_edge_left = None
        self._filt_d_edge_left = None
        self._d_dot_edge_left = 0.0

      if edge_right_valid:
        if self._prev_d_edge_right is not None:
          raw_er = float(np.clip((raw_d_edge_right - self._prev_d_edge_right) / dt_m, -MAX_EDGE_D_DOT_MPS, 0.30))
          self._d_dot_edge_right = (1.0 - alpha_ve) * self._d_dot_edge_right + alpha_ve * raw_er
        self._prev_d_edge_right = raw_d_edge_right
        if self._filt_d_edge_right is None:
          self._filt_d_edge_right = raw_d_edge_right
        else:
          self._filt_d_edge_right = (1.0 - alpha_pe) * self._filt_d_edge_right + alpha_pe * raw_d_edge_right
      else:
        self._prev_d_edge_right = None
        self._filt_d_edge_right = None
        self._d_dot_edge_right = 0.0

    if not edge_left_valid:
      self._filt_d_edge_left = None
    elif self._filt_d_edge_left is None:
      self._filt_d_edge_left = raw_d_edge_left

    if not edge_right_valid:
      self._filt_d_edge_right = None
    elif self._filt_d_edge_right is None:
      self._filt_d_edge_right = raw_d_edge_right

    d_edge_left = self._filt_d_edge_left if self._filt_d_edge_left is not None else raw_d_edge_left
    d_edge_right = self._filt_d_edge_right if self._filt_d_edge_right is not None else raw_d_edge_right

    d_dot_left = self._d_dot_left
    d_dot_right = self._d_dot_right
    d_dot_edge_left = self._d_dot_edge_left
    d_dot_edge_right = self._d_dot_edge_right

    # Time-To-Cross (TTC) for painted lines and road edges
    ttc_left = max(0.0, d_left) / abs(d_dot_left) if d_dot_left < -MIN_TTC_CLOSING_SPEED_MPS else 99.0
    ttc_right = max(0.0, d_right) / abs(d_dot_right) if d_dot_right < -MIN_TTC_CLOSING_SPEED_MPS else 99.0
    ttc_edge_left = max(0.0, d_edge_left) / abs(d_dot_edge_left) if d_dot_edge_left < -MIN_TTC_CLOSING_SPEED_MPS else 99.0
    ttc_edge_right = max(0.0, d_edge_right) / abs(d_dot_edge_right) if d_dot_edge_right < -MIN_TTC_CLOSING_SPEED_MPS else 99.0

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

    rdm_left_unstriped = (left_prob < 0.20) or unstriped_env_left
    rdm_right_unstriped = (right_prob < 0.20) or unstriped_env_right

    # Proximity-gated & debounced TTC triggers
    if left_line_visible and d_left < (d_thresh_left + ttc_prox_left) and ttc_left < ttc_thresh:
      self._ttc_left_timer += dt
    else:
      self._ttc_left_timer = 0.0

    if right_line_visible and d_right < (d_thresh_right + ttc_prox_right) and ttc_right < ttc_thresh:
      self._ttc_right_timer += dt
    else:
      self._ttc_right_timer = 0.0

    if edge_left_valid and rdm_left_unstriped and d_edge_left < (d_edge_thresh_left + ttc_edge_prox_left) and ttc_edge_left < ttc_edge_thresh_left:
      self._ttc_edge_left_timer += dt
    else:
      self._ttc_edge_left_timer = 0.0

    if edge_right_valid and rdm_right_unstriped and d_edge_right < (d_edge_thresh_right + ttc_edge_prox_right) and ttc_edge_right < ttc_edge_thresh_right:
      self._ttc_edge_right_timer += dt
    else:
      self._ttc_edge_right_timer = 0.0

    left_rdm_holding = (self._nudge_side == "left") or (self.rdm_active and self.active and self.desired_curvature > ACTIVE_CURVATURE_EPS)
    right_rdm_holding = (self._nudge_side == "right") or (self.rdm_active and self.active and self.desired_curvature < -ACTIVE_CURVATURE_EPS)
    left_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "left" and d_dot_left <= 0.0) else 0.0
    right_hyst = NUDGE_RELEASE_HYSTERESIS_M if (self._nudge_side == "right" and d_dot_right <= 0.0) else 0.0
    edge_left_hyst = (EDGE_RELEASE_HYSTERESIS_M if unstriped_env_left else 0.03) if (left_rdm_holding and d_dot_edge_left <= 0.12) else 0.0
    edge_right_hyst = (EDGE_RELEASE_HYSTERESIS_M if unstriped_env_right else 0.03) if (right_rdm_holding and d_dot_edge_right <= 0.12) else 0.0

    sustain_left_line = (
      (self._nudge_side == "left" and not self.rdm_active) and
      (d_left < d_thresh_left + ttc_prox_left) and
      (d_dot_left < -0.015)
    )
    sustain_right_line = (
      (self._nudge_side == "right" and not self.rdm_active) and
      (d_right < d_thresh_right + ttc_prox_right) and
      (d_dot_right < -0.015)
    )
    sustain_rdm_left = (
      (left_rdm_holding and self.rdm_active) and
      rdm_left_unstriped and
      (d_edge_left < d_edge_thresh_left + (0.12 if unstriped_env_left else ttc_edge_prox_left)) and
      (d_dot_edge_left < (0.01 if unstriped_env_left else -0.015))
    )
    sustain_rdm_right = (
      (right_rdm_holding and self.rdm_active) and
      rdm_right_unstriped and
      (d_edge_right < d_edge_thresh_right + (0.12 if unstriped_env_right else ttc_edge_prox_right)) and
      (d_dot_edge_right < (0.01 if unstriped_env_right else -0.015))
    )

    # 8. Candidate Trigger Detection (Left vs. Right before override cooldown gating)
    raw_trigger_left_line = (
      mode == LaneKeepMode.LANE_LINE_AND_ROAD_EDGE and
      left_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_left <= 0.0 and
      (d_left < (d_thresh_left + left_hyst) or self._ttc_left_timer >= TTC_CONFIRM_TIME_S or sustain_left_line)
    )

    raw_trigger_right_line = (
      mode == LaneKeepMode.LANE_LINE_AND_ROAD_EDGE and
      right_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_right <= 0.0 and
      (d_right < (d_thresh_right + right_hyst) or self._ttc_right_timer >= TTC_CONFIRM_TIME_S or sustain_right_line)
    )

    raw_trigger_rdm_left = (
      edge_left_valid and
      not steer_suppressed and
      self._blinker_cooldown_left <= 0.0 and
      (
        (rdm_left_unstriped and (
          d_edge_left < (d_edge_thresh_left + edge_left_hyst) or
          self._ttc_edge_left_timer >= TTC_CONFIRM_TIME_S or
          sustain_rdm_left
        )) or
        (left_line_visible and d_left <= -0.02 and d_edge_left < 0.22)
      )
    )

    raw_trigger_rdm_right = (
      edge_right_valid and
      not steer_suppressed and
      self._blinker_cooldown_right <= 0.0 and
      (
        (rdm_right_unstriped and (
          d_edge_right < (d_edge_thresh_right + edge_right_hyst) or
          self._ttc_edge_right_timer >= TTC_CONFIRM_TIME_S or
          sustain_rdm_right
        )) or
        (right_line_visible and d_right <= -0.02 and d_edge_right < 0.22)
      )
    )

    # 9. Directional Driver Override, Latched Cooldown & Lockout Evaluation
    # Maintain active override cooldowns (prevents machine-gun ON-OFF-ON pulsing against resting hands)
    if self._override_cooldown_left > 0.0:
      holding_left = (abs(steering_torque) > 0.35) if torque_unit_nm else (steering_torque > OVERRIDE_HOLD_CAN_TORQUE)
      left_returned_safe = (
        (not left_line_visible or d_left > d_thresh_left + 0.12) and
        (not edge_left_valid or not rdm_left_unstriped or d_edge_left > d_edge_thresh_left + 0.15)
      )
      if holding_left:
        self._override_cooldown_left = max(self._override_cooldown_left - dt, 0.35)
      elif left_returned_safe:
        self._override_cooldown_left = 0.0
      else:
        self._override_cooldown_left = max(0.0, self._override_cooldown_left - dt)

    if self._override_cooldown_right > 0.0:
      holding_right = (abs(steering_torque) > 0.35) if torque_unit_nm else (steering_torque < -OVERRIDE_HOLD_CAN_TORQUE)
      right_returned_safe = (
        (not right_line_visible or d_right > d_thresh_right + 0.12) and
        (not edge_right_valid or not rdm_right_unstriped or d_edge_right > d_edge_thresh_right + 0.15)
      )
      if holding_right:
        self._override_cooldown_right = max(self._override_cooldown_right - dt, 0.35)
      elif right_returned_safe:
        self._override_cooldown_right = 0.0
      else:
        self._override_cooldown_right = max(0.0, self._override_cooldown_right - dt)

    if raw_trigger_rdm_left or raw_trigger_left_line or self._nudge_side == "left" or self.desired_curvature > ACTIVE_CURVATURE_EPS:
      candidate_side = "left"
    elif raw_trigger_rdm_right or raw_trigger_right_line or self._nudge_side == "right" or self.desired_curvature < -ACTIVE_CURVATURE_EPS:
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

      if candidate_side is not None:
        self._override_confirm_timer = (self._override_confirm_timer + dt) if torque_exceeded else 0.0
        driver_override = self._override_confirm_timer >= OVERRIDE_CONFIRM_TIME_S
      else:
        self._override_confirm_timer = 0.0
        driver_override = torque_exceeded

      # Post-arrest gentle-straightening release: once outward painted-line drift has been arrested
      # and the car is receding inward (or back inside threshold), allow light driver straightening
      # pressure (> 60 CAN units) to seamlessly release the hold instead of countersteering.
      if not driver_override and self.active and not self.rdm_active and self._intervention_timer >= 0.15:
        if (
          candidate_side == "left" and
          (d_dot_left >= 0.02 or (d_left >= d_thresh_left and d_dot_left >= -0.005)) and
          steering_torque > POST_ARREST_STRAIGHTEN_CAN_TORQUE
        ):
          self._straighten_confirm_timer += dt
        elif (
          candidate_side == "right" and
          (d_dot_right >= 0.02 or (d_right >= d_thresh_right and d_dot_right >= -0.005)) and
          steering_torque < -POST_ARREST_STRAIGHTEN_CAN_TORQUE
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

    trigger_left_line = raw_trigger_left_line and (self._override_cooldown_left <= 0.0)
    trigger_rdm_left = raw_trigger_rdm_left and (self._override_cooldown_left <= 0.0)
    trigger_right_line = raw_trigger_right_line and (self._override_cooldown_right <= 0.0)
    trigger_rdm_right = raw_trigger_rdm_right and (self._override_cooldown_right <= 0.0)

    if self._lockout:
      left_clear = (not left_line_visible or d_left > d_thresh_left + 0.15) and (not edge_left_valid or d_edge_left > d_edge_thresh_left + 0.15)
      right_clear = (not right_line_visible or d_right > d_thresh_right + 0.15) and (not edge_right_valid or d_edge_right > d_edge_thresh_right + 0.15)
      # Clear lockout once driver applies torque or vehicle returns safely inside boundaries
      if driver_override or (left_clear and right_clear):
        self._lockout = False
        self._intervention_timer = 0.0
        self._rdm_timer = 0.0

    # 10. Compute Drift-Stopping Boundary Guidance (Curvature)
    target_curvature = 0.0
    is_left_nudge = False
    is_right_nudge = False
    rdm_active = False

    # Gentle drift-arresting gains: damps outward drift toward the line and fades out
    # as soon as outward drift stops (d_dot > 0) so the car parallels the line instead
    # of pushing back across the lane.
    kp = 0.0024
    kd = 0.0014

    if not driver_override and not post_arrest_release:
      if trigger_rdm_left or trigger_left_line:
        is_left_nudge = True
        rdm_active = bool(trigger_rdm_left)
        raw_b_curv = self._left_edge_curv if trigger_rdm_left else self._left_line_curv
        if not self.active:
          self.boundary_curvature = raw_b_curv
        else:
          alpha_b = float(dt / (TAU_SMOOTH + dt))
          self.boundary_curvature = (1.0 - alpha_b) * self.boundary_curvature + alpha_b * raw_b_curv
        if not self._lockout:
          d_cur = d_edge_left if trigger_rdm_left else d_left
          d_lim = (d_edge_thresh_left + edge_left_hyst) if trigger_rdm_left else (d_thresh_left + left_hyst)
          d_dot_cur = d_dot_edge_left if trigger_rdm_left else d_dot_left
          d_err = max(0.0, d_lim - d_cur)
          vel_left = max(0.0, -d_dot_cur)
          recede_ref = 0.22 if trigger_rdm_left else 0.05
          recede_scale = max(0.0, 1.0 - (d_dot_cur / recede_ref)) if d_dot_cur > 0.0 else 1.0
          # Positive curvature turns RIGHT away from left edge
          target_curvature = float((kp * d_err + kd * vel_left) * recede_scale)
      elif trigger_rdm_right or trigger_right_line:
        is_right_nudge = True
        rdm_active = bool(trigger_rdm_right)
        raw_b_curv = self._right_edge_curv if trigger_rdm_right else self._right_line_curv
        if not self.active:
          self.boundary_curvature = raw_b_curv
        else:
          alpha_b = float(dt / (TAU_SMOOTH + dt))
          self.boundary_curvature = (1.0 - alpha_b) * self.boundary_curvature + alpha_b * raw_b_curv
        if not self._lockout:
          d_cur = d_edge_right if trigger_rdm_right else d_right
          d_lim = (d_edge_thresh_right + edge_right_hyst) if trigger_rdm_right else (d_thresh_right + right_hyst)
          d_dot_cur = d_dot_edge_right if trigger_rdm_right else d_dot_right
          d_err = max(0.0, d_lim - d_cur)
          vel_right = max(0.0, -d_dot_cur)
          recede_ref = 0.22 if trigger_rdm_right else 0.05
          recede_scale = max(0.0, 1.0 - (d_dot_cur / recede_ref)) if d_dot_cur > 0.0 else 1.0
          # Negative curvature turns LEFT away from right edge
          target_curvature = -float((kp * d_err + kd * vel_right) * recede_scale)

    actively_nudging = (is_left_nudge or is_right_nudge) and abs(target_curvature) > 0.0
    if is_left_nudge and abs(target_curvature) > 0.0:
      self._nudge_side = "left"
    elif is_right_nudge and abs(target_curvature) > 0.0:
      self._nudge_side = "right"
    else:
      self._nudge_side = None

    # Clamp lateral acceleration to ceiling (0.40 m/s^2 standard, 0.50 m/s^2 RDM)
    max_accel = MAX_LAT_ACCEL * (MAX_STEER_RATE_SCALE if rdm_active else 1.0)
    max_curvature = max_accel / max(v_ego * v_ego, 1.0)
    target_curvature = float(np.clip(target_curvature, -max_curvature, max_curvature))

    # 11. Anti-Lazy & RDM Authority Timer Progression
    nudge_active = is_left_nudge or is_right_nudge
    override_active = driver_override or post_arrest_release or (self._override_cooldown_left > 0.0) or (self._override_cooldown_right > 0.0)
    if (actively_nudging or (nudge_active and self._lockout)) and not override_active and not self._lockout:
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

    self.nudge_active = bool(actively_nudging and not override_active and not self._lockout)

    # 12. Asymmetric S-Curve Envelope Smoothing Filter (gentle onset, crisp release, fast yield on override)
    if override_active:
      tau = TAU_OVERRIDE_RELEASE
    elif abs(target_curvature) > abs(self._prev_curvature) and (target_curvature * self._prev_curvature >= 0.0):
      tau = TAU_SMOOTH
    elif rdm_active or self.rdm_active:
      tau = TAU_RDM_RELEASE
    else:
      tau = TAU_RELEASE
    alpha = float(dt / (tau + dt))
    self.desired_curvature = (1.0 - alpha) * self._prev_curvature + alpha * target_curvature
    if abs(target_curvature) == 0.0 and abs(self.desired_curvature) <= ACTIVE_CURVATURE_EPS:
      self.desired_curvature = 0.0
    self._prev_curvature = self.desired_curvature
    self.desired_lat_accel = float(self.desired_curvature * (v_ego * v_ego))

    # 13. Active Actuation & Solid Cluster Line Visibility States
    self.active = bool(self.nudge_active or abs(self.desired_curvature) > ACTIVE_CURVATURE_EPS)
    if not self.active and not self.nudge_active:
      self.boundary_curvature = 0.0

    if (is_left_nudge and self.nudge_active) or (self.active and self.desired_curvature > ACTIVE_CURVATURE_EPS):
      active_side = "left"
    elif (is_right_nudge and self.nudge_active) or (self.active and self.desired_curvature < -ACTIVE_CURVATURE_EPS):
      active_side = "right"
    else:
      active_side = None

    if steer_suppressed or self._blinker_cooldown_left > 0.0 or (self._override_cooldown_left > 0.0 and not self.active):
      self._left_yellow_hold = 0.0
    elif active_side == "left":
      self._left_yellow_hold = YELLOW_LINE_HOLD_S
      self._right_yellow_hold = 0.0
    else:
      self._left_yellow_hold = max(0.0, self._left_yellow_hold - dt)

    if steer_suppressed or self._blinker_cooldown_right > 0.0 or (self._override_cooldown_right > 0.0 and not self.active):
      self._right_yellow_hold = 0.0
    elif active_side == "right":
      self._right_yellow_hold = YELLOW_LINE_HOLD_S
      self._left_yellow_hold = 0.0
    else:
      self._right_yellow_hold = max(0.0, self._right_yellow_hold - dt)

    left_yellow = (active_side == "left") or (self._left_yellow_hold > 0.0)
    right_yellow = (active_side == "right") or (self._right_yellow_hold > 0.0)

    # Suppressed during sharp cornering (>= 45 deg). Amber (3) stays solid for the full
    # active steering envelope plus a 0.40s minimum hold so 10 Hz CAN never flickers or misses a nudge.
    if steer_suppressed:
      self.left_line_state = 0
      self.right_line_state = 0
    else:
      if self._blinker_cooldown_left > 0.0:
        self.left_line_state = 0
      elif left_yellow:
        self.left_line_state = 3  # Amber / Yellow
      elif left_line_visible:
        self.left_line_state = 1  # Solid White
      else:
        self.left_line_state = 0

      if self._blinker_cooldown_right > 0.0:
        self.right_line_state = 0
      elif right_yellow:
        self.right_line_state = 3  # Amber / Yellow
      elif right_line_visible:
        self.right_line_state = 1  # Solid White
      else:
        self.right_line_state = 0

    # 14. UI Gradient State
    self.rdm_active = bool(rdm_active or (self.is_red_gradient and (left_yellow or right_yellow)))
    self.is_red_gradient = self.rdm_active

    if self.active or left_yellow or right_yellow:
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
