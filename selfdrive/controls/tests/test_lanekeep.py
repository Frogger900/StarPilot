from types import SimpleNamespace
import numpy as np
import pytest

from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.lanekeep import (
  LaneKeepController,
  LaneKeepMode,
  LaneKeepSensitivity,
  LaneKeepAlert,
  DEFAULT_VEHICLE_HALF_WIDTH,
  DEFAULT_CAMERA_OFFSET,
)


def _make_model(
  left_y=-1.8,
  right_y=1.8,
  left_prob=0.9,
  right_prob=0.9,
  edge_left_y=-2.5,
  edge_right_y=2.5,
  edge_left_std=0.1,
  edge_right_std=0.1,
  desired_curvature=0.0,
):
  xs = np.linspace(0.0, 50.0, 52)
  return SimpleNamespace(
    laneLines=[
      SimpleNamespace(x=xs, y=np.zeros_like(xs)),
      SimpleNamespace(x=xs, y=np.full_like(xs, float(left_y))),
      SimpleNamespace(x=xs, y=np.full_like(xs, float(right_y))),
      SimpleNamespace(x=xs, y=np.zeros_like(xs)),
    ],
    laneLineProbs=[0.0, float(left_prob), float(right_prob), 0.0],
    laneLineStds=[0.0, 0.1, 0.1, 0.0],
    roadEdges=[
      SimpleNamespace(x=xs, y=np.full_like(xs, float(edge_left_y))),
      SimpleNamespace(x=xs, y=np.full_like(xs, float(edge_right_y))),
    ],
    roadEdgeStds=[float(edge_left_std), float(edge_right_std)],
    action=SimpleNamespace(desiredCurvature=float(desired_curvature)),
  )


def test_disabled_and_centering_active():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.0, right_y=1.8)  # close to left line

  # 1. Mode OFF
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    mode=LaneKeepMode.OFF,
  )
  assert not lka.active
  assert lka.desired_curvature == 0.0
  assert lka.left_line_state == 0
  assert lka.right_line_state == 0

  # 2. Lat Active (Lane Centering active) -> LKA yields full authority
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=True,
    mode=LaneKeepMode.LANE_LINE_AND_ROAD_EDGE,
  )
  assert not lka.active
  assert lka.desired_curvature == 0.0
  assert lka.left_line_state == 0
  assert lka.right_line_state == 0


def test_speed_hysteresis():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.8, right_y=1.8)

  # Under min activation speed (20 mph < 25 mph default)
  lka.update(
    model,
    v_ego=20.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    min_activation_speed_mph=25.0,
  )
  assert not lka._speed_armed
  assert lka.left_line_state == 0

  # Cross above min activation speed (26 mph > 25 mph)
  lka.update(
    model,
    v_ego=26.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    min_activation_speed_mph=25.0,
  )
  assert lka._speed_armed
  assert lka.left_line_state == 1  # Solid white when armed

  # Dip slightly below min activation speed but above hysteresis floor (23 mph)
  # Floor is 25 - 4 = 21 mph
  lka.update(
    model,
    v_ego=23.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    min_activation_speed_mph=25.0,
  )
  assert lka._speed_armed

  # Dip below hysteresis floor (20 mph < 21 mph)
  lka.update(
    model,
    v_ego=20.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    min_activation_speed_mph=25.0,
  )
  assert not lka._speed_armed
  assert lka.left_line_state == 0


def test_passive_in_lane_cruising():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.8, right_y=1.8)

  for _ in range(50):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
    )

  assert not lka.active
  assert abs(lka.desired_curvature) < 1e-4
  assert lka.left_line_state == 1   # Solid white
  assert lka.right_line_state == 1  # Solid white
  assert lka.alert == LaneKeepAlert.NONE
  assert lka.gradient_opacity == 0.0


def test_drift_left_nudge_and_amber_line():
  lka = LaneKeepController()
  # w_half=0.92, camera_offset=0.04. Left tire edge at left_y - 0.96.
  # At left_y = -1.10, d_left = 1.10 - 0.96 = 0.14m (< 0.22m normal threshold)
  model = _make_model(left_y=-1.10, right_y=2.5)

  for _ in range(20):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
    )

  assert lka.active
  # Negative curvature turns steering wheel right away from left boundary
  assert lka.desired_curvature < 0.0
  # Lateral accel capped to comfortable ceiling <= 1.1 m/s^2
  assert abs(lka.desired_lat_accel) <= 1.15
  # Left line is amber, right line is solid white
  assert lka.left_line_state == 3
  assert lka.right_line_state == 1
  # UI gradient is active, amber (not red)
  assert lka.gradient_opacity > 0.0
  assert not lka.is_red_gradient
  assert lka.alert == LaneKeepAlert.NONE


def test_drift_right_nudge_and_amber_line():
  lka = LaneKeepController()
  # Right tire edge at right_y - 0.88.
  # At right_y = 1.05, d_right = 1.05 - 0.88 = 0.17m (< 0.22m normal threshold)
  model = _make_model(left_y=-2.5, right_y=1.05)

  for _ in range(20):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
    )

  assert lka.active
  # Positive curvature turns steering wheel left away from right boundary
  assert lka.desired_curvature > 0.0
  assert abs(lka.desired_lat_accel) <= 1.15
  # Left line is white, right line is amber
  assert lka.left_line_state == 1
  assert lka.right_line_state == 3
  assert lka.gradient_opacity > 0.0
  assert not lka.is_red_gradient


def test_cornering_and_blinker_gates():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.10, right_y=2.5)  # Triggers left nudge normally

  # 1. Cornering gate: wheel turned >= 45 deg suppresses lines and assist
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=46.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
  )
  assert lka.left_line_state == 0
  assert lka.right_line_state == 0

  # Reset
  lka.reset()

  # 2. Left Blinker Gate: Left line turns off, nudge suppressed
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=True,
    right_blinker=False,
    lat_active=False,
  )
  assert lka.left_line_state == 0
  assert lka.right_line_state == 1
  assert lka._blinker_cooldown_left == 2.0


def test_driver_override():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.10, right_y=2.5)

  # Run one step with no override
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
  )
  assert lka.desired_curvature < 0.0

  # Apply driver torque > 0.5 Nm
  for _ in range(50):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.8,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
    )
  # Controller smoothly decays curvature to 0
  assert abs(lka.desired_curvature) < 1e-3


def test_anti_lazy_timeout_progression():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.10, right_y=2.5)

  dt = 0.01  # 100 Hz

  # 0 to 7 seconds: Silent nudge
  for _ in range(700):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
      dt=dt,
    )
  assert lka.alert == LaneKeepAlert.NONE
  assert not lka._lockout

  # 8 to 9.9 seconds: Stage 2 prompt (KEEP_HANDS_ON_WHEEL)
  for _ in range(150):  # total 8.5s
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
      dt=dt,
    )
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL
  assert not lka._lockout

  # >= 10 seconds: Stage 3 Lockout & Cutoff
  for _ in range(200):  # total 10.5s
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
      dt=dt,
    )
  assert lka._lockout
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL
  # S-curve decays curvature to 0 during lockout
  for _ in range(100):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
      dt=dt,
    )
  assert abs(lka.desired_curvature) < 1e-3

  # Driver taps wheel (> 0.5 Nm) -> Lockout clears
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.7,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    dt=dt,
  )
  assert not lka._lockout


def test_road_edge_mitigation_and_red_gradient():
  lka = LaneKeepController()
  # Missing painted lines (prob 0.1), close road edge at -1.15m
  model = _make_model(
    left_y=-3.0,
    right_y=3.0,
    left_prob=0.1,
    right_prob=0.1,
    edge_left_y=-1.15,
    edge_right_y=2.5,
    edge_left_std=0.2,
  )

  for _ in range(20):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
    )

  assert lka.active
  assert lka.rdm_active
  assert lka.is_red_gradient
  assert lka.desired_curvature < 0.0  # Steers right away from grass/ditch


def test_sensitivity_presets():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.25, right_y=1.8)
  # d_left = 1.25 - 0.96 = 0.29m.
  # At Late (thresh 0.00m) -> should NOT trigger
  # At Normal (thresh 0.22m) -> should NOT trigger
  # At Proactive (thresh 0.38m) -> SHOULD trigger

  # 1. Late
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    sensitivity=LaneKeepSensitivity.LATE,
  )
  assert lka.left_line_state == 1
  assert lka.gradient_width == 0.15

  # 2. Normal
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    sensitivity=LaneKeepSensitivity.NORMAL,
  )
  assert lka.left_line_state == 1
  assert lka.gradient_width == 0.30

  # 3. Proactive
  for _ in range(10):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
      sensitivity=LaneKeepSensitivity.PROACTIVE,
    )
  assert lka.left_line_state == 3  # Amber triggered
  assert lka.gradient_width == 0.45


def test_get_lane_keep_visual_state():
  from openpilot.selfdrive.controls.lib.lanekeep import get_lane_keep_visual_state

  model_center = _make_model(left_y=-1.8, right_y=1.8)
  model_left = _make_model(left_y=-1.10, right_y=2.5)
  toggles = {"lane_keep_assist": True, "lane_keep_mode": 2, "lane_keep_sensitivity": 1}

  # 1. Passive centered: no nudge
  left, right, is_red, opacity, width = get_lane_keep_visual_state(
    model_center, 30.0 * CV.MPH_TO_MS, 0.0, 0.0, False, False, False, toggles
  )
  assert not left and not right and not is_red

  # 2. Centering active (MADS/AOL engaged): yields completely
  left, right, is_red, opacity, width = get_lane_keep_visual_state(
    model_left, 30.0 * CV.MPH_TO_MS, 0.0, 0.0, False, False, True, toggles
  )
  assert not left and not right

  # 3. Drift left in manual mode: triggers amber left nudge
  for _ in range(15):
    left, right, is_red, opacity, width = get_lane_keep_visual_state(
      model_left, 30.0 * CV.MPH_TO_MS, 0.0, 0.0, False, False, False, toggles
    )
  assert left and not right
  assert not is_red  # Amber for painted line
  assert opacity > 0.0
  assert width == 0.30
