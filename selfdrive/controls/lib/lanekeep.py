import math
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
MAX_LAT_ACCEL = 1.1                # m/s^2 gentle acceleration ceiling
MAX_STEER_RATE_SCALE = 1.3         # +30% authority for emergency road edge mitigation
OVERRIDE_TORQUE_THRESHOLD = 0.5    # Nm (light hand pressure yields assist)
CORNERING_GATE_ANGLE_DEG = 45.0    # degrees (suppress assist during active turns)
APEX_CUT_BUFFER_M = 0.25           # meters (expand buffer when clipping inside curve)
POST_BLINKER_COOLDOWN_S = 2.0      # seconds to wait after turn signal auto-cancels

# S-Curve Smoothing Filter (100ms time constant)
TAU_SMOOTH = 0.10


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
    self._prev_v_ego = 0.0
    self._intervention_timer = 0.0
    self._rdm_timer = 0.0
    self._lockout = False
    self._blinker_cooldown_left = 0.0
    self._blinker_cooldown_right = 0.0
    self._speed_armed = False

  def reset(self):
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
    self._intervention_timer = 0.0
    self._rdm_timer = 0.0
    self._lockout = False

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
  ):
    """
    Core LKA / RDM cycle executed at controls rate (100 Hz / DT_CTRL).
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
      ttc_thresh = 0.90
      self.gradient_width = 0.15
    elif sensitivity == LaneKeepSensitivity.PROACTIVE:
      d_thresh = 0.38
      ttc_thresh = 1.60
      self.gradient_width = 0.45
    else:  # NORMAL (Default)
      d_thresh = 0.22
      ttc_thresh = 1.30
      self.gradient_width = 0.30

    d_edge_thresh = 0.32
    ttc_edge_thresh = 1.50

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

    # Distance to tire outer edges
    d_left = -left_y - (self.w_half + self.camera_offset)
    d_right = right_y - (self.w_half - self.camera_offset)
    d_edge_left = -edge_left_y - (self.w_half + self.camera_offset)
    d_edge_right = edge_right_y - (self.w_half - self.camera_offset)

    # Lateral velocities (d_dot)
    d_dot_left = (d_left - self._prev_d_left) / dt if self._prev_d_left is not None else 0.0
    d_dot_right = (d_right - self._prev_d_right) / dt if self._prev_d_right is not None else 0.0
    self._prev_d_left = d_left
    self._prev_d_right = d_right

    # Time-To-Cross (TTC)
    ttc_left = d_left / abs(d_dot_left) if d_dot_left < -0.05 else 99.0
    ttc_right = d_right / abs(d_dot_right) if d_dot_right < -0.05 else 99.0

    # Curve Apex Smoothing (expand boundary on inside of curve)
    road_curv = float(model_v2.action.desiredCurvature) if hasattr(model_v2, "action") else 0.0
    d_thresh_left = d_thresh + (APEX_CUT_BUFFER_M if road_curv > 0.001 else 0.0)
    d_thresh_right = d_thresh + (APEX_CUT_BUFFER_M if road_curv < -0.001 else 0.0)

    # 8. Trigger Detection (Left vs. Right)
    trigger_left_line = (
      mode == LaneKeepMode.LANE_LINE_AND_ROAD_EDGE and
      left_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_left <= 0.0 and
      (d_left < d_thresh_left or ttc_left < ttc_thresh)
    )

    trigger_right_line = (
      mode == LaneKeepMode.LANE_LINE_AND_ROAD_EDGE and
      right_line_visible and
      not steer_suppressed and
      self._blinker_cooldown_right <= 0.0 and
      (d_right < d_thresh_right or ttc_right < ttc_thresh)
    )

    # Road Edge Mitigation (RDM) Fallback
    trigger_rdm_left = (
      edge_left_std < 0.50 and
      not steer_suppressed and
      self._blinker_cooldown_left <= 0.0 and
      (d_edge_left < d_edge_thresh or (d_left < 0.0 and d_edge_left < 0.50))
    )

    trigger_rdm_right = (
      edge_right_std < 0.50 and
      not steer_suppressed and
      self._blinker_cooldown_right <= 0.0 and
      (d_edge_right < d_edge_thresh or (d_right < 0.0 and d_edge_right < 0.50))
    )

    # 9. Driver Override & Lockout Evaluation
    # steering_pressed is openpilot's calibrated boolean from vehicle CAN (e.g. > 80-150 CAN torque units).
    # If steering_pressed is not available or False, fallback to torque check if scaled.
    driver_override = steering_pressed or (abs(steering_torque) > 120.0 if abs(steering_torque) > 5.0 else abs(steering_torque) > OVERRIDE_TORQUE_THRESHOLD)

    if self._lockout:
      # Clear lockout once driver applies torque or vehicle returns to center
      if driver_override or (d_left > d_thresh_left + 0.15 and d_right > d_thresh_right + 0.15):
        self._lockout = False
        self._intervention_timer = 0.0

    # 10. Compute Corrective Spring (Curvature)
    target_curvature = 0.0
    is_left_nudge = False
    is_right_nudge = False
    rdm_active = False

    # Proportional and Derivative spring gains
    kp = 0.0065
    kd = 0.0030

    if not self._lockout and not driver_override:
      if trigger_rdm_left or trigger_left_line:
        is_left_nudge = True
        rdm_active = trigger_rdm_left
        d_err = d_edge_thresh - d_edge_left if trigger_rdm_left else d_thresh_left - d_left
        d_err = max(0.0, d_err)
        # Negative curvature turns RIGHT away from left edge
        target_curvature = -float(kp * d_err + kd * abs(d_dot_left))
      elif trigger_rdm_right or trigger_right_line:
        is_right_nudge = True
        rdm_active = trigger_rdm_right
        d_err = d_edge_thresh - d_edge_right if trigger_rdm_right else d_thresh_right - d_right
        d_err = max(0.0, d_err)
        # Positive curvature turns LEFT away from right edge
        target_curvature = float(kp * d_err + kd * abs(d_dot_right))

    # Clamp lateral acceleration to ceiling (0.8 - 1.2 m/s^2)
    max_accel = MAX_LAT_ACCEL * (MAX_STEER_RATE_SCALE if rdm_active else 1.0)
    max_curvature = max_accel / max(v_ego * v_ego, 1.0)
    target_curvature = float(np.clip(target_curvature, -max_curvature, max_curvature))

    # 11. 10-Second Anti-Lazy Timer Progression
    nudge_active = is_left_nudge or is_right_nudge
    if nudge_active and not driver_override:
      self._intervention_timer += dt
      if rdm_active:
        self._rdm_timer += dt
      else:
        self._rdm_timer = 0.0
    else:
      self._intervention_timer = max(0.0, self._intervention_timer - 2.0 * dt)
      self._rdm_timer = max(0.0, self._rdm_timer - 2.0 * dt)

    # Alert Stage Evaluation
    if rdm_active and self._rdm_timer >= 15.0:
      # RDM Emergency: approaching ditch timeout
      self.alert = LaneKeepAlert.EMERGENCY_DEPARTURE
    elif self._intervention_timer >= 10.0:
      # Stage 3: Cutoff & Lockout (prevent hands-off abuse)
      self._lockout = True
      target_curvature = 0.0
      self.alert = LaneKeepAlert.KEEP_HANDS_ON_WHEEL
    elif self._intervention_timer >= 8.0:
      # Stage 2: Approaching timeout prompt
      self.alert = LaneKeepAlert.KEEP_HANDS_ON_WHEEL
    else:
      # Stage 1: Silent gentle nudge
      self.alert = LaneKeepAlert.NONE

    # 12. S-Curve Envelope Smoothing Filter
    alpha = float(dt / (TAU_SMOOTH + dt))
    self.desired_curvature = (1.0 - alpha) * self._prev_curvature + alpha * target_curvature
    self._prev_curvature = self.desired_curvature
    self.desired_lat_accel = float(self.desired_curvature * (v_ego * v_ego))

    # 13. Cluster Line Visibility States
    # Suppressed during sharp cornering (>= 45 deg)
    if steer_suppressed:
      self.left_line_state = 0
      self.right_line_state = 0
    else:
      if self._blinker_cooldown_left > 0.0 or not left_line_visible:
        self.left_line_state = 0
      elif is_left_nudge:
        self.left_line_state = 3  # Amber / Yellow
      else:
        self.left_line_state = 1  # Solid White

      if self._blinker_cooldown_right > 0.0 or not right_line_visible:
        self.right_line_state = 0
      elif is_right_nudge:
        self.right_line_state = 3  # Amber / Yellow
      else:
        self.right_line_state = 1  # Solid White

    # 14. UI Gradient State
    self.active = abs(self.desired_curvature) > 1e-5
    self.rdm_active = rdm_active
    self.is_red_gradient = rdm_active

    if self.active:
      # Opacity scales dynamically with proximity
      proximity = max(0.0, min(1.0, abs(self.desired_lat_accel) / max_accel))
      self.gradient_opacity = float(0.35 + 0.65 * proximity)
    else:
      self.gradient_opacity = 0.0


_ui_lka_controller = None

def _get_toggle(toggles, key, default):
  if isinstance(toggles, dict):
    return toggles.get(key, default)
  return getattr(toggles, key, default)

def get_lane_keep_visual_state(
  model_v2,
  v_ego: float,
  steering_angle_deg: float,
  steering_torque: float,
  *args,
) -> tuple[bool, bool, bool, float, float]:
  """
  Returns (is_left_nudge, is_right_nudge, is_red, opacity, ribbon_width).
  """
  global _ui_lka_controller
  if _ui_lka_controller is None:
    _ui_lka_controller = LaneKeepController()

  if len(args) == 4:
    left_blinker, right_blinker, lat_active, toggles = args
    steering_pressed = False
  elif len(args) >= 5:
    steering_pressed, left_blinker, right_blinker, lat_active, toggles = args[:5]
  else:
    return False, False, False, 0.0, 0.30

  enabled = bool(_get_toggle(toggles, "lane_keep_assist", True))
  if not enabled or lat_active:
    _ui_lka_controller.reset()
    return False, False, False, 0.0, 0.30

  mode = int(_get_toggle(toggles, "lane_keep_mode", LaneKeepMode.LANE_LINE_AND_ROAD_EDGE))
  sensitivity = int(_get_toggle(toggles, "lane_keep_sensitivity", LaneKeepSensitivity.NORMAL))
  min_speed = float(_get_toggle(toggles, "lane_keep_min_speed", 25.0))

  _ui_lka_controller.update(
    model_v2=model_v2,
    v_ego=v_ego,
    steering_angle_deg=steering_angle_deg,
    steering_torque=steering_torque,
    left_blinker=left_blinker,
    right_blinker=right_blinker,
    lat_active=lat_active,
    steering_pressed=steering_pressed,
    mode=mode,
    sensitivity=sensitivity,
    min_activation_speed_mph=min_speed,
  )

  is_left = _ui_lka_controller.left_line_state == 3
  is_right = _ui_lka_controller.right_line_state == 3
  return (
    is_left,
    is_right,
    _ui_lka_controller.is_red_gradient,
    _ui_lka_controller.gradient_opacity,
    _ui_lka_controller.gradient_width,
  )
