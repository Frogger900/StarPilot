import math
from types import SimpleNamespace
import numpy as np
import pytest

from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR as SUBARU_CAR
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.lanekeep import (
  LaneKeepController,
  LaneKeepAlert,
  extract_lane_keep_visual_state,
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

  # 1. Master LKA disabled
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    enabled=False,
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
    enabled=True,
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

  # Dip slightly below min activation speed but above hysteresis floor (23 mph > 21 mph)
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


def test_drift_left_nudge_and_vehicle_model_right_steer_sign():
  lka = LaneKeepController()
  CP = CarInterface.get_non_essential_params(SUBARU_CAR.SUBARU_LEGACY_2025)
  VM = VehicleModel(CP)
  v_ego = 30.0 * CV.MPH_TO_MS

  # w_half=0.92, camera_offset=0.04. At left_y = -1.00, d_left = 1.00 - 0.96 = 0.04m (< 0.07m standard threshold)
  model = _make_model(left_y=-1.00, right_y=2.5)

  for _ in range(20):
    lka.update(
      model,
      v_ego=v_ego,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
    )

  assert lka.active
  assert lka.desired_curvature > 0.0
  angle_steers_des = math.degrees(VM.get_steer_from_curvature(-lka.desired_curvature, v_ego, 0.0))
  assert angle_steers_des < 0.0
  assert abs(lka.desired_lat_accel) <= 0.45
  assert lka.left_line_state == 3
  assert lka.right_line_state == 1
  assert lka.gradient_opacity > 0.0
  assert lka.alert == LaneKeepAlert.NONE


def test_drift_right_nudge_and_vehicle_model_left_steer_sign():
  lka = LaneKeepController()
  CP = CarInterface.get_non_essential_params(SUBARU_CAR.SUBARU_LEGACY_2025)
  VM = VehicleModel(CP)
  v_ego = 30.0 * CV.MPH_TO_MS

  # At right_y = 0.92, d_right = 0.92 - 0.88 = 0.04m (< 0.07m standard threshold)
  model = _make_model(left_y=-2.5, right_y=0.92)

  for _ in range(20):
    lka.update(
      model,
      v_ego=v_ego,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
    )

  assert lka.active
  assert lka.desired_curvature < 0.0
  angle_steers_des = math.degrees(VM.get_steer_from_curvature(-lka.desired_curvature, v_ego, 0.0))
  assert angle_steers_des > 0.0
  assert abs(lka.desired_lat_accel) <= 0.45
  assert lka.left_line_state == 1
  assert lka.right_line_state == 3
  assert lka.gradient_opacity > 0.0


def test_inside_curve_apex_cut_buffer():
  v_ego = 35.0 * CV.MPH_TO_MS

  # 1. Sweeping LEFT curve (desired_curvature = -0.005 < -0.001):
  # Inside left boundary threshold shrinks by 0.12m (0.07 -> -0.05m), so d_left = 0.04m does NOT nuisance-nudge
  lka_left_curve = LaneKeepController()
  model_left_apex = _make_model(left_y=-1.00, right_y=2.5, desired_curvature=-0.005)
  for _ in range(15):
    lka_left_curve.update(model_left_apex, v_ego=v_ego, steering_angle_deg=8.0, steering_torque=0.0)
  assert not lka_left_curve.active
  assert lka_left_curve.left_line_state == 1

  # Outside right boundary on the same left curve retains full 0.07m margin, so d_right = 0.04m DOES nudge left!
  model_left_outside = _make_model(left_y=-2.5, right_y=0.92, desired_curvature=-0.005)
  for _ in range(15):
    lka_left_curve.update(model_left_outside, v_ego=v_ego, steering_angle_deg=8.0, steering_torque=0.0)
  assert lka_left_curve.active
  assert lka_left_curve.desired_curvature < 0.0
  assert lka_left_curve.right_line_state == 3

  # 2. Sweeping RIGHT curve (desired_curvature = +0.005 > +0.001):
  # Inside right boundary threshold shrinks by 0.12m (0.07 -> -0.05m), so d_right = 0.04m does NOT nuisance-nudge
  lka_right_curve = LaneKeepController()
  model_right_apex = _make_model(left_y=-2.5, right_y=0.92, desired_curvature=0.005)
  for _ in range(15):
    lka_right_curve.update(model_right_apex, v_ego=v_ego, steering_angle_deg=-8.0, steering_torque=0.0)
  assert not lka_right_curve.active
  assert lka_right_curve.right_line_state == 1


def test_cornering_and_blinker_gates():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.00, right_y=2.5)  # Triggers left nudge normally

  # 1. Cornering gate: wheel turned >= 35 deg suppresses lines and assist
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=36.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
  )
  assert lka.left_line_state == 0
  assert lka.right_line_state == 0

  lka.reset(full=True)

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
  model = _make_model(left_y=-1.00, right_y=2.5)

  # Run one step with no override
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
    left_blinker=False,
    right_blinker=False,
    lat_active=False,
    torque_unit_nm=True,
  )
  assert lka.desired_curvature > 0.0

  # Apply driver torque > 0.5 Nm on Nm-scaled platform
  for _ in range(50):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.8,
      left_blinker=False,
      right_blinker=False,
      lat_active=False,
      torque_unit_nm=True,
    )
  # Controller smoothly decays curvature to 0
  assert abs(lka.desired_curvature) < 1e-3


def test_anti_lazy_timeout_progression():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.00, right_y=2.5)
  dt = 0.01  # 100 Hz

  # 0 to 4.5 seconds: Silent nudge (< 5.0s initial warning threshold)
  for _ in range(450):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      dt=dt,
    )
  assert lka.alert == LaneKeepAlert.NONE
  assert not lka._lockout

  # >= 5.0 seconds (5.5s total): Stage 2 prompt (KEEP_HANDS_ON_WHEEL), still steering up to 15.0s cutoff
  for _ in range(100):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      dt=dt,
    )
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL
  assert not lka._lockout
  assert lka.active

  # At 14.0 seconds (< 15.0s cutoff): Still prompting KEEP_HANDS_ON_WHEEL while steering
  for _ in range(850):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      dt=dt,
    )
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL
  assert not lka._lockout

  # >= 15.0 seconds (15.2s total): Stage 3 Lockout & Cutoff -> EMERGENCY_DEPARTURE ("Lane Keep Assist Timed Out")
  for _ in range(120):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      dt=dt,
    )
  assert lka._lockout
  assert lka.alert == LaneKeepAlert.EMERGENCY_DEPARTURE

  # S-curve decays curvature to 0 during lockout while keeping EMERGENCY_DEPARTURE active
  for _ in range(100):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      dt=dt,
    )
  assert abs(lka.desired_curvature) < 1e-3
  assert lka.alert == LaneKeepAlert.EMERGENCY_DEPARTURE

  # Driver applies intentional hands-on wheel torque (> 140 CAN units) -> Lockout and 30s window clear
  for _ in range(6):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=155.0,
      dt=dt,
    )
  assert not lka._lockout
  assert lka.alert == LaneKeepAlert.NONE


def test_anti_lazy_hands_off_ping_pong_and_30s_rolling_window():
  v_ego = 35.0 * CV.MPH_TO_MS
  dt = 0.01
  left_bounce = _make_model(left_y=-1.00, right_y=2.55)
  right_bounce = _make_model(left_y=-2.55, right_y=0.92)
  centered = _make_model(left_y=-1.80, right_y=1.80)

  # Part 1: Hands-off lane-to-lane ping-ponging (3 x 1.7s bounces = 5.1s active within < 15s)
  lka = LaneKeepController()

  def step_frames(m, n, torque=0.0):
    for i in range(n):
      lka.update(m, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=torque, dt=dt, model_updated=(i % 5 == 0))

  # Bounce 1 (left, 1.7s) + coast across lane center (3.0s)
  step_frames(left_bounce, 170)
  assert lka.alert == LaneKeepAlert.NONE
  step_frames(centered, 300)
  assert lka.alert == LaneKeepAlert.NONE

  # Bounce 2 (right, 1.7s -> ~3.4s cumulative) + coast across lane center (3.0s)
  step_frames(right_bounce, 170)
  assert lka.alert == LaneKeepAlert.NONE
  step_frames(centered, 300)

  # Bounce 3 (left, 1.7s -> ~5.1s cumulative within 30s window!) -> KEEP_HANDS_ON_WHEEL triggers!
  step_frames(left_bounce, 170)
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL
  assert not lka._lockout

  # Coasting hands-off across lane center does NOT clear KEEP_HANDS_ON_WHEEL!
  step_frames(centered, 450)
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL
  assert not lka._lockout

  # Bounce 4 (right, 1.5s) + coast (4.5s) -> >= 10.0s hands-off grace timer after initial 5s warning (15s total)!
  step_frames(right_bounce, 150)
  step_frames(centered, 450)

  # Next boundary touch latches lockout + cutoff + elevated EMERGENCY_DEPARTURE alert!
  step_frames(left_bounce, 30)
  assert lka._lockout
  assert lka.alert == LaneKeepAlert.EMERGENCY_DEPARTURE

  # Coasting hands-off back into lane center does NOT auto-reset lockout!
  step_frames(centered, 200, torque=0.0)
  assert lka._lockout
  assert lka.alert == LaneKeepAlert.EMERGENCY_DEPARTURE

  # Actual driver hands-on steering torque (>= 85 CAN units idle) clears lockout and resets the 30s window!
  step_frames(centered, 15, torque=95.0)
  assert not lka._lockout
  assert lka.alert == LaneKeepAlert.NONE

  # Part 2: 30-second sliding window aging — occasional brief 2.0s nudges separated by 29s never false-warn
  lka2 = LaneKeepController()
  for i in range(200):  # 2.0s nudge at t=0..2s
    lka2.update(left_bounce, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=dt, model_updated=(i % 5 == 0))
  for i in range(2900):  # 29.0s centered driving at t=2..31s (ages out first nudge past 30s window)
    lka2.update(centered, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=dt, model_updated=(i % 5 == 0))
  for i in range(200):  # 2.0s nudge at t=31..33s -> window sum stays <= 2.1s (< 5.0s)
    lka2.update(right_bounce, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=dt, model_updated=(i % 5 == 0))
  assert lka2.alert == LaneKeepAlert.NONE
  assert not lka2._lockout


def test_extract_lane_keep_visual_state():
  # 1. None or disarmed -> inactive 4-tuple (is_left, is_right, opacity, width)
  assert extract_lane_keep_visual_state(None) == (False, False, 0.0, 0.30)
  assert extract_lane_keep_visual_state(SimpleNamespace(lkaArmed=False, lkaLeftLineState=3)) == (
    False, False, 0.0, 0.30
  )

  # 2. Armed passive standby -> no ribbon
  passive = SimpleNamespace(
    lkaArmed=True,
    lkaLeftLineState=1,
    lkaRightLineState=1,
    lkaGradientOpacity=0.0,
    lkaGradientWidth=0.30,
  )
  assert extract_lane_keep_visual_state(passive) == (False, False, 0.0, 0.30)

  # 3. Left painted-line nudge -> Amber highlight
  left_nudge = SimpleNamespace(
    lkaArmed=True,
    lkaLeftLineState=3,
    lkaRightLineState=1,
    lkaGradientOpacity=0.72,
    lkaGradientWidth=0.30,
  )
  assert extract_lane_keep_visual_state(left_nudge) == (True, False, pytest.approx(0.72), pytest.approx(0.30))

  # 4. Right painted-line nudge -> Amber highlight
  right_nudge = SimpleNamespace(
    lkaArmed=True,
    lkaLeftLineState=1,
    lkaRightLineState=3,
    lkaGradientOpacity=0.88,
    lkaGradientWidth=0.30,
  )
  assert extract_lane_keep_visual_state(right_nudge) == (False, True, pytest.approx(0.88), pytest.approx(0.30))


def test_road_edges_never_trigger_steering_or_yellow_lines():
  """Unstriped road edges or close curbs/shoulders never trigger LKA (painted lane lines only)."""
  lka = LaneKeepController()
  v_ego = 30.0 * CV.MPH_TO_MS

  # 1. Unstriped road with very close physical road edge (edge_left_y = -0.95 -> 0.0m clearance)
  unstriped = _make_model(
    left_y=-1.30, right_y=2.20, left_prob=0.10, right_prob=0.05,
    edge_left_y=-0.95, edge_right_y=2.50, edge_left_std=0.05, edge_right_std=0.05,
  )
  for step in range(50):
    lka.update(unstriped, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
    assert not lka.active
    assert lka.desired_curvature == 0.0
    assert lka.left_line_state == 0
    assert lka.right_line_state == 0

  # 2. Narrow striped road where roadEdge hugs the fog line while car is safely inside painted lines
  narrow_striped = _make_model(
    left_y=-1.49, right_y=1.26, left_prob=0.85, right_prob=0.85,
    edge_left_y=-1.55, edge_right_y=0.92, edge_left_std=0.08, edge_right_std=0.08,
  )
  for step in range(50):
    lka.update(narrow_striped, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
    assert not lka.active
    assert lka.desired_curvature == 0.0
    assert lka.left_line_state == 1
    assert lka.right_line_state == 1


def test_yellow_line_minimum_one_second_hold_no_flicker():
  """Threshold touches and quick driver recoveries keep the yellow line solid for >= 1.0s without flickering."""
  v_ego = 38.0 * CV.MPH_TO_MS
  dt = 0.01
  touch_left = _make_model(left_y=-1.01, right_y=2.50)  # d_left = 0.05m < 0.07m
  receded = _make_model(left_y=-1.45, right_y=2.50)     # d_left = 0.49m (well inside lane)

  # Case 1: Brief 0.10s threshold touch followed immediately by receding back into lane
  lka = LaneKeepController()
  yellow_frames = 0
  for step in range(10):  # 0.10s touch
    lka.update(touch_left, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=dt, model_updated=(step % 5 == 0))
    assert lka.left_line_state == 3
    yellow_frames += 1

  # Next 0.95s (95 frames -> 1.05s total from start, >= 0.95s after touch ended): must stay solid 3 every single frame!
  for step in range(95):
    lka.update(receded, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=dt, model_updated=(step % 5 == 0))
    assert lka.left_line_state == 3
    yellow_frames += 1
  assert yellow_frames >= 105  # >= 1.05s solid yellow with zero flicker!

  # After hold expires (~1.25s total), returns cleanly to solid white (1)
  for step in range(30):
    lka.update(receded, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=dt, model_updated=(step % 5 == 0))
  assert lka.left_line_state == 1

  # Case 2: Post-arrest override cooldown does NOT truncate an already-latched 1.0s yellow hold early
  lka2 = LaneKeepController()
  for step in range(20):  # 0.20s active nudge
    lka2.update(touch_left, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=dt, model_updated=(step % 5 == 0))
  assert lka2.left_line_state == 3
  receding_near = _make_model(left_y=-1.04, right_y=2.50)
  for step in range(95):  # 0.95s of inward driver recovery torque (-95 CAN units) -> post_arrest_release fires
    lka2.update(receding_near, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=-95.0, dt=dt, model_updated=(step % 5 == 0))
    assert lka2.left_line_state == 3


def test_20hz_vision_jitter_at_100hz_ctrl_never_false_triggers_mid_lane_or_far_road_edge():
  """Regression test for route 00000102--9dbe33ca26:
  100 Hz controls loop reading 20 Hz modelV2 with realistic 3-4 cm line jitter and 18 cm far road-edge
  jitter while driving centered in the lane must produce ZERO false nudges or yellow line flashes.
  """
  lka = LaneKeepController()
  v_ego = 38.0 * CV.MPH_TO_MS
  rng = np.random.default_rng(42)

  cur_model = _make_model(left_y=-1.58, right_y=1.68, edge_left_y=-2.35, edge_right_y=2.55, edge_right_std=0.30)
  for step in range(500):  # 5.0 seconds at 100 Hz
    model_updated = (step % 5) == 0
    if model_updated:
      cur_model = _make_model(
        left_y=-1.58 + float(rng.uniform(-0.04, 0.04)),
        right_y=1.68 + float(rng.uniform(-0.04, 0.04)),
        edge_left_y=-2.35 + float(rng.uniform(-0.18, 0.18)),
        edge_right_y=2.55 + float(rng.uniform(-0.18, 0.18)),
        edge_left_std=0.28,
        edge_right_std=0.32,
      )
    lka.update(
      cur_model,
      v_ego=v_ego,
      steering_angle_deg=0.0,
      steering_torque=25.0,
      dt=0.01,
      model_updated=model_updated,
    )
    assert not lka.active
    assert lka.desired_curvature == 0.0
    assert lka.left_line_state == 1
    assert lka.right_line_state == 1


def test_gentle_accel_ceiling_and_clean_release_without_sticky_tail():
  lka = LaneKeepController()
  v_ego = 45.0 * CV.MPH_TO_MS
  deep_drift = _make_model(left_y=-0.94, right_y=2.6)
  for _ in range(60):
    lka.update(deep_drift, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01)
  assert lka.active
  # Standard LKA lateral acceleration stays <= 0.40 m/s^2 (gentle boundary-parallel arrest ceiling)
  assert abs(lka.desired_lat_accel) <= 0.405

  # Intentional opposing CAN hand torque (155 > 140 opposing threshold) releases active EPS actuation cleanly
  for _ in range(25):
    lka.update(deep_drift, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=155.0, dt=0.01)
  assert not lka.active
  assert lka.desired_curvature == 0.0
  # Override cooldown remains latched so releasing hands slightly doesn't immediately re-pulse
  for _ in range(20):
    lka.update(deep_drift, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=40.0, dt=0.01)
  assert not lka.active


def test_passive_resting_hands_and_zero_crossing_can_torque_never_pulse():
  """Regression test for route 00000103--cbac80378f:
  1) Small integer CAN zero-crossing torque (+/-1..+/-5) must NOT fall into an Nm heuristic and trip override.
  2) Passive resting-hand torsion-bar drag (40..125 CAN units opposing, up to 170 assisting/curve-holding)
     while outward drift is still active must keep LKA actuation smooth with zero ON->OFF->ON drops.
  """
  lka = LaneKeepController()
  v_ego = 40.0 * CV.MPH_TO_MS
  left_drift = _make_model(left_y=-1.00, right_y=2.55)
  torque_trace = [0.0, 1.0, 3.0, -2.0, -4.0, 5.0, 55.0, 85.0, 115.0, 125.0, 72.0, -165.0, 3.0, 95.0]

  for idx in range(len(torque_trace) * 5):
    trq = torque_trace[idx // 5]
    lka.update(
      left_drift,
      v_ego=v_ego,
      steering_angle_deg=0.0,
      steering_torque=trq,
      dt=0.01,
      model_updated=(idx % 5 == 0),
      torque_unit_nm=False,
    )
    if idx >= 5:
      assert lka.active
      assert lka.desired_curvature > 0.0
      assert lka.left_line_state == 3


def test_boundary_curvature_drift_stopping_fade_and_yellow_hold():
  lka = LaneKeepController()
  v_ego = 38.0 * CV.MPH_TO_MS
  xs = np.linspace(0.0, 50.0, 52)
  curved_left_y = -1.00 + 0.001 * (xs ** 2)
  curved_model = _make_model(left_y=-1.00, right_y=2.55)
  curved_model.laneLines[1] = SimpleNamespace(x=xs, y=curved_left_y)

  for step in range(25):
    lka.update(
      curved_model,
      v_ego=v_ego,
      steering_angle_deg=0.0,
      steering_torque=0.0,
      dt=0.01,
      model_updated=(step % 5 == 0),
    )
  assert lka.active
  assert lka.boundary_curvature == pytest.approx(0.002, abs=1e-4)
  assert lka.left_line_state == 3
  peak_curv = lka.desired_curvature
  assert peak_curv > 0.0

  base_y = -1.00
  for _ in range(6):
    base_y -= 0.009  # +0.18 m/s d_left increase per 50ms modelV2 step
    rec_model = _make_model(left_y=base_y, right_y=2.55)
    rec_model.laneLines[1] = SimpleNamespace(x=xs, y=base_y + 0.001 * (xs ** 2))
    for sub in range(5):
      lka.update(
        rec_model,
        v_ego=v_ego,
        steering_angle_deg=0.0,
        steering_torque=0.0,
        dt=0.01,
        model_updated=(sub == 0),
      )
      if lka.active:
        assert lka.left_line_state == 3

  assert lka.desired_curvature < peak_curv * 0.5
  assert lka.left_line_state == 3


def test_crisp_post_arrest_release_straightening_and_inward_recovery_torque():
  """Regression test for routes 00000104..00000106:
  1) Passive inward recession back across d_thresh=0.07m releases cleanly without sticky hold.
  2) Gentle driver straightening torque (> 60 CAN units) once drift begins receding releases the hold.
  3) Inward driver recovery torque (> 80 CAN units turning back toward lane center once drift has stopped)
     immediately releases LKA so LKA never stacks on top of the driver's own recovery steer.
  """
  v_ego = 40.0 * CV.MPH_TO_MS
  drift_model = _make_model(left_y=-1.00, right_y=2.55)  # d_left = 0.04m < 0.07m

  # Part 1: Passive inward recession
  lka = LaneKeepController()
  for step in range(25):
    lka.update(drift_model, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  assert lka.active and lka.nudge_active

  for ly in (-1.02, -1.05, -1.08):
    rec_m = _make_model(left_y=ly, right_y=2.55)
    for sub in range(5):
      lka.update(rec_m, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(sub == 0))
  assert not lka.nudge_active

  safe_m = _make_model(left_y=-1.08, right_y=2.55)
  for step in range(30):
    lka.update(safe_m, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  assert not lka.active
  assert lka.desired_curvature == 0.0

  # Part 2: Gentle driver straightening torque (+75 CAN units opposing right turn) once receding
  lka2 = LaneKeepController()
  for step in range(25):
    lka2.update(drift_model, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  receding_near_line = _make_model(left_y=-1.015, right_y=2.55)
  for sub in range(6):
    lka2.update(receding_near_line, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=75.0, dt=0.01, model_updated=(sub == 0))
  assert not lka2.nudge_active

  # Part 3: Inward driver recovery torque (-95 CAN units turning right away from left line) once receding
  lka3 = LaneKeepController()
  for step in range(25):
    lka3.update(drift_model, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  assert lka3.active and lka3.nudge_active
  for sub in range(6):
    lka3.update(receding_near_line, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=-95.0, dt=0.01, model_updated=(sub == 0))
  assert not lka3.nudge_active
  for _ in range(20):
    lka3.update(receding_near_line, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=-95.0, dt=0.01, model_updated=False)
  assert not lka3.active
  assert lka3.desired_curvature == 0.0


def test_pre_existing_high_driver_torque_prevents_40ms_onset_flash_and_hold():
  """Regression test for route 0000010f--64946bed87 (Episodes 01, 06, 10, 11, 12):
  When the driver is already holding high steering torque (> 185 CAN units curve-holding or > 140 opposing)
  for >= 50 ms BEFORE crossing a boundary threshold, override is already confirmed so LKA never flashes
  active for 40 ms or latches a 1.05s yellow line on touch.
  """
  lka = LaneKeepController()
  v_ego = 35.0 * CV.MPH_TO_MS
  centered = _make_model(left_y=-1.80, right_y=1.80)
  touch_left = _make_model(left_y=-0.99, right_y=2.55)

  # Driver holds +210 CAN units for 0.10s before approaching left line
  for step in range(10):
    lka.update(
      centered,
      v_ego=v_ego,
      steering_angle_deg=25.0,
      steering_torque=210.0,
      dt=0.01,
      model_updated=(step % 5 == 0),
      torque_unit_nm=False,
    )
    assert not lka.active
    assert lka.left_line_state == 1

  # Car touches left line while driver is still holding +210 CAN units: zero 40ms micro-activation!
  for step in range(20):
    lka.update(
      touch_left,
      v_ego=v_ego,
      steering_angle_deg=28.0,
      steering_torque=210.0,
      dt=0.01,
      model_updated=(step % 5 == 0),
      torque_unit_nm=False,
    )
    assert not lka.active
    assert not lka.nudge_active
    assert lka.desired_curvature == 0.0
    assert lka.left_line_state == 1


def test_ttc_trigger_sustains_until_receding_and_cornering_gate_preserves_latched_yellow_hold():
  """Regression test for route 0000010f--64946bed87:
  1) A TTC-triggered nudge at d_right = 0.11m (> d_thresh = 0.07m) remains continuously active while
     outward drift slows to 0.0 m/s until the car actually begins receding (d_dot >= 0.015 m/s).
  2) Crossing the 45-deg cornering gate mid-hold stops new nudges and hides white lines (0),
     while allowing an already-latched 1.05s amber line (3) to finish its countdown cleanly.
  """
  v_ego = 52.5 * CV.MPH_TO_MS
  lka = LaneKeepController()

  # Part 1: Approach right line from d_right = 0.20m down to 0.11m (TTC trigger), then hold at 0.11m
  for ry in (1.08, 1.05, 1.02, 0.99):
    m = _make_model(left_y=-2.60, right_y=ry)
    for sub in range(5):
      lka.update(m, v_ego=v_ego, steering_angle_deg=2.8, steering_torque=25.0, dt=0.01, model_updated=(sub == 0))
  assert lka.active and lka.nudge_active
  assert lka.right_line_state == 3

  # Hold parallel at right_y = 0.99 (d_right = 0.11m, d_dot -> 0.0 m/s): must NOT drop out prematurely!
  hold_m = _make_model(left_y=-2.60, right_y=0.99)
  for step in range(25):
    lka.update(hold_m, v_ego=v_ego, steering_angle_deg=3.2, steering_torque=22.0, dt=0.01, model_updated=(step % 5 == 0))
    assert lka.nudge_active
    assert lka.desired_curvature < 0.0
    assert lka.right_line_state == 3

  # Once car recedes inward (right_y = 1.01 -> 1.04 -> 1.08), nudge_active releases cleanly
  for ry in (1.01, 1.04, 1.08):
    rec_m = _make_model(left_y=-2.60, right_y=ry)
    for sub in range(5):
      lka.update(rec_m, v_ego=v_ego, steering_angle_deg=3.0, steering_torque=15.0, dt=0.01, model_updated=(sub == 0))
  assert not lka.nudge_active

  # Part 2: Latched yellow line hold continues for full >= 1.0s even if wheel crosses >= 35 deg mid-hold
  lka2 = LaneKeepController()
  touch_left = _make_model(left_y=-1.00, right_y=2.50)
  for step in range(20):  # 0.20s active nudge at 28 deg (< 35 deg gate)
    lka2.update(touch_left, v_ego=32.0 * CV.MPH_TO_MS, steering_angle_deg=28.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  assert lka2.left_line_state == 3
  assert lka2.right_line_state == 1

  # Wheel turns to 38 deg (>= 35 deg cornering gate): new nudge stops, right white line hides (0),
  # but already-latched left amber hold (3) finishes its remaining ~1.0s countdown without truncation!
  for step in range(85):  # 0.85s at 38 deg
    lka2.update(touch_left, v_ego=32.0 * CV.MPH_TO_MS, steering_angle_deg=38.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
    assert not lka2.nudge_active
    assert lka2.left_line_state == 3
    assert lka2.right_line_state == 0

  # After hold countdown finishes, left line also hides (0) while >= 35 deg
  for step in range(35):
    lka2.update(touch_left, v_ego=32.0 * CV.MPH_TO_MS, steering_angle_deg=38.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  assert lka2.left_line_state == 0
  assert lka2.right_line_state == 0


def test_single_line_reduced_sensitivity_road_edge_pinch_and_cornering_unwind_cooldown():
  """Regression test for route 00000111--05b42a314b:
  1) When only one painted line is visible, hugging the yellow line at d_left = +0.05m does not nuisance-trigger,
     while crossing onto/over the line (d_left = -0.04m) triggers cleanly.
  2) When the opposite painted line is missing and the physical roadEdge pinches tightly, the road-edge
     narrow-lane pinch guard backs off d_thresh down to -0.06m so LKA never pushes into a tight shoulder.
  3) Unwinding from >= 35 deg holds a 0.50s post-cornering cooldown before re-enabling LKA triggers.
  """
  v_ego = 38.0 * CV.MPH_TO_MS
  lka = LaneKeepController()

  # 1. Single visible left line with left tire at d_left = +0.05m (left_y = -1.01) -> no trigger
  hug_single_left = _make_model(
    left_y=-1.01, right_y=1.80, left_prob=0.90, right_prob=0.10,
    edge_left_y=-2.50, edge_right_y=1.85, edge_right_std=0.15,
  )
  for step in range(40):
    lka.update(hug_single_left, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
    assert not lka.active
    assert lka.left_line_state == 1
    assert lka.right_line_state == 0

  # Crossing onto the single left line (left_y = -0.92 -> d_left = -0.04m < -0.02m) with ample right road edge -> triggers!
  cross_single_left = _make_model(
    left_y=-0.92, right_y=1.80, left_prob=0.90, right_prob=0.10,
    edge_left_y=-2.50, edge_right_y=1.85, edge_right_std=0.15,
  )
  for step in range(20):
    lka.update(cross_single_left, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  assert lka.active
  assert lka.desired_curvature > 0.0
  assert lka.left_line_state == 3

  # 2. Tight right road-edge pinch (edge_right_y = 1.15 -> d_right_edge = 1.15 - 0.88 - 0.25 = +0.02m)
  # pushes half_clearance down so d_left = -0.04m does NOT nudge toward the right ditch/edge!
  lka.reset(full=True)
  tight_edge_pinch = _make_model(
    left_y=-0.92, right_y=1.80, left_prob=0.90, right_prob=0.10,
    edge_left_y=-2.50, edge_right_y=1.15, edge_right_std=0.15,
  )
  for step in range(30):
    lka.update(tight_edge_pinch, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
    assert not lka.active
    assert lka.desired_curvature == 0.0

  # 3. 35-deg cornering gate + 0.50s unwind cooldown
  lka.reset(full=True)
  two_line_cross = _make_model(left_y=-0.98, right_y=2.50)
  for step in range(10):
    lka.update(two_line_cross, v_ego=v_ego, steering_angle_deg=38.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
    assert not lka.active
    assert lka.left_line_state == 0

  # Unwind wheel to 25 deg: suppressed for the first 0.45s of the 0.50s unwind cooldown
  for step in range(45):
    lka.update(two_line_cross, v_ego=v_ego, steering_angle_deg=25.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
    assert not lka.active

  # After 0.50s unwind cooldown expires, LKA resumes normally
  for step in range(15):
    lka.update(two_line_cross, v_ego=v_ego, steering_angle_deg=25.0, steering_torque=0.0, dt=0.01, model_updated=(step % 5 == 0))
  assert lka.active
  assert lka.left_line_state == 3


