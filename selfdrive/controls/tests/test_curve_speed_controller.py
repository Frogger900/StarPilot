import numpy as np
import pytest

from types import SimpleNamespace

from openpilot.common.realtime import DT_MDL
from openpilot.starpilot.common.starpilot_variables import DEFAULT_LATERAL_ACCELERATION, PLANNER_TIME
from openpilot.starpilot.controls.lib.curve_speed_controller import (
  CSC_APPROACH_DECEL,
  CSC_COUNT_CAP,
  CSC_EGO_HEADROOM,
  CSC_MIN_SPEED,
  CSC_NUDGE_WEIGHT,
  CurveSpeedController,
)


class FakeParams:
  def __init__(self, values=None):
    self.values = dict(values or {})

  def get(self, *args, **kwargs):
    key = args[0] if args else None
    return self.values.get(key)

  def put_nonblocking(self, key, value):
    self.values[key] = value


def make_controller(curve_profile=None, curvature_data=None, weather_id=0, reduce_lat=0.0, road_curvature=0.02, driving_in_curve=False):
  if curve_profile is None:
    curve_profile = (np.zeros(33), np.linspace(0.0, 300.0, 33))

  planner = SimpleNamespace(
    params=FakeParams({"CurvatureData": curvature_data} if curvature_data is not None else None),
    curve_profile=curve_profile,
    starpilot_weather=SimpleNamespace(weather_id=weather_id, reduce_lateral_acceleration=reduce_lat),
    road_curvature=road_curvature,
    driving_in_curve=driving_in_curve,
    tracking_lead=False,
    lateral_acceleration=0.0,
  )
  controller = CurveSpeedController(SimpleNamespace(starpilot_planner=planner))
  return planner, controller


def make_sm(*, gas=False, brake=False, long_active=True, blinker=False, accel_pressed=False):
  return {
    "carControl": SimpleNamespace(longActive=long_active),
    "carState": SimpleNamespace(gasPressed=gas, brakePressed=brake, leftBlinker=blinker, rightBlinker=False),
    "starpilotCarState": SimpleNamespace(accelPressed=accel_pressed),
    "onroadEvents": [],
  }


def single_apex_profile(curvature, distance):
  distances = np.linspace(0.0, max(distance * 1.5, 1.0), 33)
  curvatures = np.zeros(33)
  index = int(np.argmin(np.abs(distances - distance)))
  distances[index] = distance
  curvatures[index] = curvature
  return curvatures, distances


def converge(controller, v_ego, v_cruise, frames=600):
  for _ in range(frames):
    controller.update_target(v_ego, v_cruise)
  return controller.target


def envelope_speed(controller, curvature, distance):
  curve_speed = max(float(np.sqrt(controller.lat_accel_for_curvature(curvature) / curvature)), CSC_MIN_SPEED)
  return float(np.sqrt(curve_speed**2 + 2.0 * CSC_APPROACH_DECEL * distance))


def test_straight_road_target_is_cruise_speed():
  _, controller = make_controller()

  controller.update_target(30.0, 30.0)

  assert controller.target == pytest.approx(30.0)


def test_distant_apex_does_not_constrain_until_braking_is_due():
  _, controller = make_controller(curve_profile=single_apex_profile(0.02, 400.0))

  target = converge(controller, 30.0, 30.0)

  assert target == pytest.approx(30.0)


def test_apex_in_braking_range_constrains_to_kinematic_envelope():
  _, controller = make_controller(curve_profile=single_apex_profile(0.02, 150.0))

  target = converge(controller, 30.0, 30.0)

  assert target == pytest.approx(envelope_speed(controller, 0.02, 150.0), abs=0.1)
  assert target < 30.0


def test_exit_recovery_rises_immediately_without_freeze():
  planner, controller = make_controller(curve_profile=single_apex_profile(0.03, 20.0))
  low_target = converge(controller, 15.0, 30.0)
  assert low_target < 20.0

  # once the envelope releases, the ego floor lifts the target within one frame
  # so re-acceleration is never dragged behind the car
  planner.curve_profile = (np.zeros(33), np.linspace(0.0, 300.0, 33))
  controller.update_target(15.0, 30.0)
  assert controller.target == pytest.approx(15.0 + CSC_EGO_HEADROOM)

  recovered = converge(controller, 15.0, 30.0)
  assert recovered == pytest.approx(30.0)


def test_fresh_activation_seeds_at_envelope_not_cruise():
  _, controller = make_controller(curve_profile=(np.full(33, 0.05), np.linspace(0.0, 60.0, 33)))

  controller.update_target(6.0, 30.0)

  # launching into a sharp turn: the target starts near the envelope (floored at
  # ego speed + headroom), not at v_cruise with a multi-second ramp down
  assert controller.target < 15.0


def test_target_never_trails_accelerating_car_when_unconstrained():
  planner, controller = make_controller(curve_profile=single_apex_profile(0.03, 20.0))
  converge(controller, 15.0, 30.0)

  planner.curve_profile = (np.zeros(33), np.linspace(0.0, 300.0, 33))
  v_ego = 15.0
  for _ in range(100):
    v_ego = min(v_ego + 2.0 * DT_MDL, 30.0)
    controller.update_target(v_ego, 30.0)
    assert controller.target >= min(30.0, v_ego + CSC_EGO_HEADROOM) - 1e-6


def test_target_does_not_ratchet_down_with_ego_speed():
  _, controller = make_controller(curve_profile=single_apex_profile(0.02, 150.0))
  target = converge(controller, 30.0, 30.0)
  assert target > 15.0

  controller.update_target(14.0, 30.0)

  assert controller.target == pytest.approx(target, abs=0.2)


def test_sharp_curve_target_floors_at_min_speed():
  _, controller = make_controller(curve_profile=(np.full(33, 0.1), np.linspace(0.0, 100.0, 33)))

  target = converge(controller, 15.0, 30.0)

  assert target == pytest.approx(CSC_MIN_SPEED, abs=0.05)


def test_weather_reduces_curve_speed():
  _, dry = make_controller(curve_profile=single_apex_profile(0.01, 0.0))
  _, wet = make_controller(curve_profile=single_apex_profile(0.01, 0.0), weather_id=1, reduce_lat=0.2)

  dry_target = converge(dry, 20.0, 30.0)
  wet_target = converge(wet, 20.0, 30.0)

  assert wet_target < dry_target
  assert wet_target == pytest.approx(dry_target * np.sqrt(0.8), abs=0.1)


def test_prior_gives_higher_lat_accel_for_sharper_curves():
  _, controller = make_controller()

  assert controller.lat_accel_for_curvature(0.001) == pytest.approx(1.5, abs=0.05)
  assert controller.lat_accel_for_curvature(0.1) == pytest.approx(2.9, abs=0.05)
  assert controller.lateral_acceleration == pytest.approx(DEFAULT_LATERAL_ACCELERATION)


def test_heavily_sampled_bucket_dominates_prior():
  _, controller = make_controller(curvature_data={"0.05": {"average": 3.0, "count": 100000}})

  assert controller.lat_accel_for_curvature(0.05) == pytest.approx(3.0, abs=0.05)
  assert controller.lat_accel_for_curvature(0.08) >= controller.lat_accel_for_curvature(0.05)


def test_learned_curve_stays_monotonic_despite_low_outlier_bucket():
  _, controller = make_controller(curvature_data={"0.05": {"average": 0.5, "count": 100000}})

  assert controller.lat_accel_for_curvature(0.05) >= controller.lat_accel_for_curvature(0.03)


def test_legacy_off_grid_curvature_data_merges_into_buckets():
  _, controller = make_controller(curvature_data={
    "0.0203": {"average": 2.5, "count": 10},
    "0.02": {"average": 2.0, "count": 10},
  })

  assert controller.curvature_data["0.02"]["count"] == 20
  assert controller.curvature_data["0.02"]["average"] == pytest.approx(2.25)


def test_training_update_step_is_capped_by_ema_count():
  planner, controller = make_controller(curvature_data={"0.02": {"average": 2.0, "count": 10000}}, driving_in_curve=True)
  planner.lateral_acceleration = 3.0
  controller.training_timer = PLANNER_TIME

  controller.log_data(10.0, make_sm(long_active=False))

  data = controller.curvature_data["0.02"]
  assert data["count"] == 10001
  assert data["average"] == pytest.approx((2.0 * CSC_COUNT_CAP + 3.0) / (CSC_COUNT_CAP + 1))


def test_no_passive_training_right_after_csc_limited_speed():
  planner, controller = make_controller(curve_profile=single_apex_profile(0.03, 20.0), driving_in_curve=True)
  planner.lateral_acceleration = 3.0
  converge(controller, 15.0, 30.0)
  assert controller.training_quiet_timer > 0.0

  controller.training_timer = PLANNER_TIME
  controller.log_data(10.0, make_sm(long_active=False))
  assert "0.02" not in controller.curvature_data
  assert not controller.enable_training

  controller.training_quiet_timer = 0.0
  controller.training_timer = PLANNER_TIME
  controller.log_data(10.0, make_sm(long_active=False))
  assert controller.curvature_data["0.02"]["count"] == 1


def test_gas_override_nudges_bucket_up_once_per_episode():
  _, controller = make_controller()
  prior = controller.lat_accel_for_curvature(0.02)
  controller.target = 10.0

  controller.handle_override(20.0, True, make_sm(gas=True))
  assert controller.curvature_data["0.02"]["count"] == CSC_NUDGE_WEIGHT
  assert controller.curvature_data["0.02"]["average"] > prior

  controller.handle_override(20.0, True, make_sm(gas=True))
  assert controller.curvature_data["0.02"]["count"] == CSC_NUDGE_WEIGHT

  controller.handle_override(20.0, False, make_sm())
  controller.target = 10.0
  controller.handle_override(20.0, True, make_sm(gas=True))
  assert controller.curvature_data["0.02"]["count"] == 2 * CSC_NUDGE_WEIGHT


def test_res_button_nudges_bucket_up_even_at_target_speed():
  _, controller = make_controller()
  prior = controller.lat_accel_for_curvature(0.02)
  controller.target = 20.0  # car tracking the target, so the gas-press condition would not fire

  controller.handle_override(20.0, True, make_sm(), accel_button=True)

  assert controller.curvature_data["0.02"]["count"] == CSC_NUDGE_WEIGHT
  assert controller.curvature_data["0.02"]["average"] > prior


def test_brake_override_nudges_bucket_down():
  _, controller = make_controller(driving_in_curve=True)
  prior = controller.lat_accel_for_curvature(0.02)

  controller.handle_override(20.0, True, make_sm(brake=True))

  assert controller.curvature_data["0.02"]["count"] == CSC_NUDGE_WEIGHT
  assert controller.curvature_data["0.02"]["average"] < prior


def test_calibrated_lateral_acceleration_param_is_written_on_flush():
  planner, controller = make_controller(curvature_data={"0.02": {"average": 2.8, "count": 5000}})

  # writes are debounced through the dirty flag, so nothing lands until a flush
  assert "CalibratedLateralAcceleration" not in planner.params.values
  controller.data_dirty = True
  controller.flush_data()

  assert planner.params.values["CalibratedLateralAcceleration"] > DEFAULT_LATERAL_ACCELERATION
  assert controller.lateral_acceleration == planner.params.values["CalibratedLateralAcceleration"]
