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
  LaneKeepMode,
  LaneKeepSensitivity,
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

  # w_half=0.92, camera_offset=0.04. At left_y = -1.00, d_left = 1.00 - 0.96 = 0.04m (< 0.07m normal threshold)
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
  # Openpilot convention: +desired_curvature (> 0) turns RIGHT; LatControlAngle (latcontrol_angle.py:68)
  # passes -desired_curvature to VehicleModel, yielding negative steeringAngleDeg (< 0, clockwise / right turn on Subaru).
  assert lka.desired_curvature > 0.0
  angle_steers_des = math.degrees(VM.get_steer_from_curvature(-lka.desired_curvature, v_ego, 0.0))
  assert angle_steers_des < 0.0
  assert abs(lka.desired_lat_accel) <= 0.45
  assert lka.left_line_state == 3
  assert lka.right_line_state == 1
  assert lka.gradient_opacity > 0.0
  assert not lka.is_red_gradient
  assert lka.alert == LaneKeepAlert.NONE


def test_drift_right_nudge_and_vehicle_model_left_steer_sign():
  lka = LaneKeepController()
  CP = CarInterface.get_non_essential_params(SUBARU_CAR.SUBARU_LEGACY_2025)
  VM = VehicleModel(CP)
  v_ego = 30.0 * CV.MPH_TO_MS

  # At right_y = 0.92, d_right = 0.92 - 0.88 = 0.04m (< 0.07m normal threshold)
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
  # Openpilot convention: -desired_curvature (< 0) turns LEFT; LatControlAngle (latcontrol_angle.py:68)
  # passes -desired_curvature to VehicleModel, yielding positive steeringAngleDeg (> 0, counter-clockwise / left turn on Subaru).
  assert lka.desired_curvature < 0.0
  angle_steers_des = math.degrees(VM.get_steer_from_curvature(-lka.desired_curvature, v_ego, 0.0))
  assert angle_steers_des > 0.0
  assert abs(lka.desired_lat_accel) <= 0.45
  assert lka.left_line_state == 1
  assert lka.right_line_state == 3
  assert lka.gradient_opacity > 0.0
  assert not lka.is_red_gradient


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

  # 0 to 7 seconds: Silent nudge
  for _ in range(700):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=0.0,
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
      dt=dt,
    )
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL
  assert not lka._lockout

  # >= 10 seconds: Stage 3 Lockout & Cutoff -> EMERGENCY_DEPARTURE persists throughout lockout
  for _ in range(200):  # total 10.5s
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

  # Driver applies intentional wheel torque (> 140 CAN units) -> Lockout clears
  for _ in range(6):
    lka.update(
      model,
      v_ego=30.0 * CV.MPH_TO_MS,
      steering_angle_deg=0.0,
      steering_torque=155.0,
      dt=dt,
    )
  assert not lka._lockout


def test_road_edge_mitigation_ttc_amber_cluster_and_18s_authority():
  lka = LaneKeepController()
  v_ego = 30.0 * CV.MPH_TO_MS

  # 1. Proactive unstriped road-edge TTC trigger (missing painted lines, d_edge_left shrinking rapidly within 0.28m zone)
  m1 = _make_model(left_prob=0.1, right_prob=0.1, edge_left_y=-1.26, edge_right_y=2.5, edge_left_std=0.2)
  m2 = _make_model(left_prob=0.1, right_prob=0.1, edge_left_y=-1.16, edge_right_y=2.5, edge_left_std=0.2)
  lka.update(m1, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.10)
  lka.update(m2, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.10)
  assert lka.active
  assert lka.rdm_active
  assert lka.is_red_gradient
  assert lka.left_line_state == 3  # Amber cluster line even on unstriped road!
  assert lka.desired_curvature > 0.0  # Steers RIGHT away from left ditch

  # 2. Close road edge sustained intervention: RDM keeps steering past 10s up to 18s ceiling (urgent alert at 15s)
  lka.reset()
  model_edge = _make_model(
    left_y=-3.0, right_y=3.0, left_prob=0.1, right_prob=0.1,
    edge_left_y=-1.04, edge_right_y=2.5, edge_left_std=0.2,
  )
  for _ in range(1200):  # 12.0s (> 10s LKA cutoff, < 15s RDM emergency warning)
    lka.update(model_edge, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01)
  assert not lka._lockout
  assert lka.active
  assert lka.desired_curvature > 0.0
  assert lka.alert == LaneKeepAlert.KEEP_HANDS_ON_WHEEL

  for _ in range(350):  # 15.5s (>= 15s RDM emergency warning, still actively steering!)
    lka.update(model_edge, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01)
  assert not lka._lockout
  assert lka.active
  assert lka.alert == LaneKeepAlert.EMERGENCY_DEPARTURE

  for _ in range(300):  # 18.5s (>= 18s RDM lockout ceiling)
    lka.update(model_edge, v_ego=v_ego, steering_angle_deg=0.0, steering_torque=0.0, dt=0.01)
  assert lka._lockout
  assert lka.alert == LaneKeepAlert.EMERGENCY_DEPARTURE


def test_sensitivity_presets():
  lka = LaneKeepController()
  model = _make_model(left_y=-1.10, right_y=1.8)
  # d_left = 1.10 - 0.96 = 0.14m: passive on LATE (-0.03m) and NORMAL (0.07m), triggers PROACTIVE (0.18m)

  # 1. Late
  lka.update(
    model,
    v_ego=30.0 * CV.MPH_TO_MS,
    steering_angle_deg=0.0,
    steering_torque=0.0,
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
      sensitivity=LaneKeepSensitivity.PROACTIVE,
    )
  assert lka.left_line_state == 3
  assert lka.gradient_width == 0.45


def test_extract_lane_keep_visual_state():
  # 1. None or disarmed -> inactive
  assert extract_lane_keep_visual_state(None) == (False, False, False, 0.0, 0.30)
  assert extract_lane_keep_visual_state(SimpleNamespace(lkaArmed=False, lkaLeftLineState=3)) == (
    False, False, False, 0.0, 0.30
  )

  # 2. Armed passive standby -> no ribbon
  passive = SimpleNamespace(
    lkaArmed=True,
    lkaLeftLineState=1,
    lkaRightLineState=1,
    lkaRdmActive=False,
    lkaGradientOpacity=0.0,
    lkaGradientWidth=0.30,
  )
  assert extract_lane_keep_visual_state(passive) == (False, False, False, 0.0, 0.30)

  # 3. Left painted-line nudge -> Amber ribbon
  left_nudge = SimpleNamespace(
    lkaArmed=True,
    lkaLeftLineState=3,
    lkaRightLineState=1,
    lkaRdmActive=False,
    lkaGradientOpacity=0.72,
    lkaGradientWidth=0.30,
  )
  assert extract_lane_keep_visual_state(left_nudge) == (True, False, False, pytest.approx(0.72), pytest.approx(0.30))

  # 4. Right road-edge RDM nudge (Proactive width) -> Neon Red ribbon with 0.45m width
  right_rdm = SimpleNamespace(
    lkaArmed=True,
    lkaLeftLineState=0,
    lkaRightLineState=3,
    lkaRdmActive=True,
    lkaGradientOpacity=0.95,
    lkaGradientWidth=0.45,
  )
  assert extract_lane_keep_visual_state(right_rdm) == (False, True, True, pytest.approx(0.95), pytest.approx(0.45))


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


def test_visible_painted_line_prevents_mid_lane_rdm_false_trigger():
  """Regression test for route 00000103--cbac80378f:
  On a narrow ~2.75m local road where grass/asphalt edge hugs the white fog line (d_edge_right ~ 0.10-0.20m)
  while the car is safely centered 0.38m inside the visible painted line (even if prob dips to 0.30),
  road-edge RDM must NOT fire over top of the painted line.
  """
  lka = LaneKeepController()
  v_ego = 38.0 * CV.MPH_TO_MS
  for prob in (0.85, 0.45, 0.30, 0.25):
    # d_left = 1.49 - 0.96 = 0.53m, d_right = 1.26 - 0.88 = 0.38m, d_edge_right = 0.98 - 0.88 = 0.10m (< 0.12m!)
    narrow_road = _make_model(
      left_y=-1.49,
      right_y=1.26,
      left_prob=0.85,
      right_prob=prob,
      edge_left_y=-1.65,
      edge_right_y=0.98,
      edge_left_std=0.12,
      edge_right_std=0.12,
    )
    for step in range(40):
      lka.update(
        narrow_road,
        v_ego=v_ego,
        steering_angle_deg=0.0,
        steering_torque=0.0,
        dt=0.01,
        model_updated=(step % 5 == 0),
      )
      assert not lka.active
      assert not lka.rdm_active
      assert lka.desired_curvature == 0.0
      assert lka.right_line_state in (0, 1)


def test_passive_resting_hands_and_zero_crossing_can_torque_never_pulse():
  """Regression test for route 00000103--cbac80378f:
  1) Small integer CAN zero-crossing torque (+/-1..+/-5) must NOT fall into an Nm heuristic and trip override.
  2) Passive resting-hand torsion-bar drag (40..125 CAN units opposing, up to 170 assisting/curve-holding)
     must keep LKA actuation and yellow line state completely smooth with zero ON->OFF->ON drops.
  """
  lka = LaneKeepController()
  v_ego = 40.0 * CV.MPH_TO_MS
  # Left-line nudge: d_left = 1.00 - 0.96 = 0.04m (< 0.07m normal threshold)
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
  """Regression test for route 00000103--cbac80378f:
  - Extracts spatial curvature of the approached painted line so controlsd aligns parallel to the boundary.
  - Fades extra push to zero once outward drift stops and begins receding (d_dot >= 0.10 m/s).
  - Keeps yellow cluster line state (3) solid across the entire active release envelope + 0.40s minimum hold.
  """
  lka = LaneKeepController()
  v_ego = 38.0 * CV.MPH_TO_MS
  xs = np.linspace(0.0, 50.0, 52)
  # Curved left line y(x) = -1.00 + 0.001 * x^2 -> spatial curvature d2y/dx2 = +0.002 (gentle right curve)
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

  # Now car recedes away from the left line at +0.18 m/s while still near d_left ~ 0.05..0.09m:
  # outward drift has stopped, so LKA fades its inward push instead of shoving toward lane center,
  # while keeping left_line_state == 3 solid throughout the active tail and 0.40s hold window!
  base_y = -1.00
  for m_frame in range(6):
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
