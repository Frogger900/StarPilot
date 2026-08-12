#!/usr/bin/env python3
import numpy as np

from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

from openpilot.starpilot.common.starpilot_variables import CITY_SPEED_LIMIT, CRUISING_SPEED, DEFAULT_LATERAL_ACCELERATION, PLANNER_TIME

CALIBRATION_PROGRESS_THRESHOLD = 10 / DT_MDL
CSC_MIN_SPEED = CITY_SPEED_LIMIT * CV.MPH_TO_MS

# Distance-resolved speed envelope. Braking for a curve ahead is planned at
# CSC_APPROACH_DECEL; the target recovers toward v_cruise as constraint points
# pass behind the car, rate-limited so curve exit reads as a smooth pull.
CSC_APPROACH_DECEL = 1.0
CSC_TARGET_UP_RATE = 3.0
CSC_TARGET_DOWN_RATE = 2.5
CSC_TARGET_FILTER_RC = 0.4
# While the envelope itself allows it, the target must never trail below the
# car's own speed — CSC only caps for curves, it must not drag re-acceleration
CSC_EGO_HEADROOM = 2.0
# Hysteresis on "CSC is limiting" so telemetry/UI don't flicker on model noise
CSC_ACTIVE_ON_DELTA = 0.5
CSC_ACTIVE_OFF_DELTA = 0.25

# Learning. Buckets keep the legacy {curvature: {average, count}} JSON schema.
CSC_COUNT_CAP = 600               # EMA floor: samples beyond this stop shrinking the update step
CSC_PRIOR_COUNT = 100             # bucket count at which learned data and the prior have equal weight
CSC_LAT_ACCEL_MIN = 1.2
CSC_LAT_ACCEL_MAX = 3.2
CSC_NUDGE = 0.15                  # m/s^2 shift applied by a driver override pseudo-sample
CSC_NUDGE_WEIGHT = 20             # counts a single override pseudo-sample is worth
CSC_TRAINING_QUIET_TIME = 5.0     # no passive samples this soon after CSC limited speed (avoid learning our own cap)

MAX_CURVATURE = 0.1
MIN_CURVATURE = 0.001
ROUNDING_PRECISION = 5
STEP = 0.001

# Comfortable lateral acceleration by curvature: drivers tolerate more in sharp
# low-speed corners than in highway sweepers. DEFAULT_LATERAL_ACCELERATION sits
# mid-range so an untrained controller behaves close to the old global default.
PRIOR_CURVATURE_BP = [0.001, 0.003, 0.01, 0.03, 0.1]
PRIOR_LAT_ACCEL_V = [1.5, 1.8, 2.2, 2.6, 2.9]


def is_user_overriding_longitudinal(sm):
  try:
    if any(getattr(event, "overrideLongitudinal", False) for event in sm["onroadEvents"]):
      return True
  except (KeyError, TypeError):
    pass

  car_state = sm["carState"]
  starpilot_car_state = sm["starpilotCarState"]
  return bool(
    getattr(car_state, "gasPressed", False) or
    getattr(car_state, "brakePressed", False) or
    getattr(starpilot_car_state, "accelPressed", False)
  )


def is_manual_speed_control(sm):
  """Return whether the driver, rather than longitudinal control, owns speed."""
  return not bool(sm["carControl"].longActive) or is_user_overriding_longitudinal(sm)


class CurveSpeedController:
  def __init__(self, StarPilotVCruise):
    self.starpilot_planner = StarPilotVCruise.starpilot_planner

    self.enable_training = False
    self.nudge_applied = False

    self.training_timer = 0.0
    self.persistence_timer = 0.0
    self.training_quiet_timer = 0.0
    self.data_dirty = False

    self.target = 0.0
    self.target_filter = FirstOrderFilter(0.0, CSC_TARGET_FILTER_RC, DT_MDL, initialized=False)
    self.seed_pending = True

    self._long_active_prev = False

    curvature_data = self.starpilot_planner.params.get("CurvatureData")
    self.curvature_data = self._normalize_curvature_data(curvature_data)

    self.required_curvatures = [str(round(road_curvature, ROUNDING_PRECISION)) for road_curvature in np.arange(MIN_CURVATURE, MAX_CURVATURE + STEP, STEP)]

    self.rebuild_lat_accel_curve()

  @staticmethod
  def _bucket_curvature(road_curvature):
    clipped_curvature = float(np.clip(road_curvature, MIN_CURVATURE, MAX_CURVATURE))
    bucket_index = round((clipped_curvature - MIN_CURVATURE) / STEP)
    bucketed_curvature = MIN_CURVATURE + (bucket_index * STEP)
    return str(round(bucketed_curvature, ROUNDING_PRECISION))

  @classmethod
  def _normalize_curvature_data(cls, curvature_data):
    if not isinstance(curvature_data, dict):
      return {}

    normalized = {}
    for key, value in curvature_data.items():
      if not isinstance(value, dict):
        continue

      try:
        raw_curvature = abs(float(key))
        average = float(value["average"])
        count = int(value["count"])
      except (KeyError, TypeError, ValueError):
        continue

      if count <= 0:
        continue

      bucket = cls._bucket_curvature(raw_curvature)
      if bucket in normalized:
        existing = normalized[bucket]
        total_count = existing["count"] + count
        normalized[bucket] = {
          "average": ((existing["average"] * existing["count"]) + (average * count)) / total_count,
          "count": total_count,
        }
      else:
        normalized[bucket] = {
          "average": average,
          "count": count,
        }

    return normalized

  def _persist_data(self):
    if not self.data_dirty:
      return

    progress = 0.0
    for key in self.required_curvatures:
      if key in self.curvature_data:
        progress += min(self.curvature_data[key]["count"] / CALIBRATION_PROGRESS_THRESHOLD, 1.0)

    self.starpilot_planner.params.put_nonblocking("CalibratedLateralAcceleration", self.lateral_acceleration)
    self.starpilot_planner.params.put_nonblocking("CalibrationProgress", (progress / len(self.required_curvatures)) * 100)
    self.starpilot_planner.params.put_nonblocking("CurvatureData", self.curvature_data)
    self.data_dirty = False
    self.persistence_timer = 0.0

  def flush_data(self):
    self._persist_data()

  def log_data(self, v_ego, sm):
    self.training_quiet_timer = max(self.training_quiet_timer - DT_MDL, 0.0)

    eligible = (
      v_ego > CRUISING_SPEED and
      not self.starpilot_planner.tracking_lead and
      is_manual_speed_control(sm) and
      self.training_quiet_timer <= 0.0
    )
    self.enable_training = False

    if not eligible:
      self.flush_data()
      self.training_timer = 0.0
      self.persistence_timer = 0.0
      return

    self.training_timer += DT_MDL
    if self.data_dirty:
      self.persistence_timer += DT_MDL

    in_curve = (
      self.training_timer >= PLANNER_TIME and
      self.starpilot_planner.driving_in_curve and
      not (sm["carState"].leftBlinker or sm["carState"].rightBlinker)
    )
    if in_curve:
      lateral_acceleration = abs(self.starpilot_planner.lateral_acceleration)
      road_curvature = self._bucket_curvature(abs(self.starpilot_planner.road_curvature))

      if road_curvature in self.curvature_data:
        data = self.curvature_data[road_curvature]
        # cap the effective count so a long-established bucket still tracks a
        # change in driving style instead of freezing on its first few hundred samples
        effective_count = min(data["count"], CSC_COUNT_CAP)
        self.curvature_data[road_curvature] = {
          "average": ((data["average"] * effective_count) + lateral_acceleration) / (effective_count + 1),
          "count": data["count"] + 1
        }
      else:
        self.curvature_data[road_curvature] = {
          "average": lateral_acceleration,
          "count": 1
        }

      self.data_dirty = True
      self.rebuild_lat_accel_curve()
      self.enable_training = True

      if self.persistence_timer >= PLANNER_TIME:
        self.flush_data()
    elif self.data_dirty:
      self.flush_data()

  def handle_override(self, v_ego, was_controlling, sm, accel_button=False):
    long_active = bool(sm["carControl"].longActive)
    long_dropped = self._long_active_prev and not long_active
    self._long_active_prev = long_active

    if not was_controlling:
      self.nudge_applied = False
      return

    if self.nudge_applied:
      return

    if accel_button or (sm["carState"].gasPressed and self.target < v_ego - 0.5):
      self._apply_nudge(CSC_NUDGE)
    elif (getattr(sm["carState"], "brakePressed", False) or long_dropped) and self.starpilot_planner.driving_in_curve:
      self._apply_nudge(-CSC_NUDGE)

  def _apply_nudge(self, offset):
    key = self._bucket_curvature(abs(self.starpilot_planner.road_curvature))
    blended = float(np.interp(float(key), self._curve_k, self._curve_a))
    sample = float(np.clip(blended + offset, CSC_LAT_ACCEL_MIN, CSC_LAT_ACCEL_MAX))

    data = self.curvature_data.get(key, {"average": sample, "count": 0})
    effective_count = min(data["count"], CSC_COUNT_CAP)
    total = effective_count + CSC_NUDGE_WEIGHT
    self.curvature_data[key] = {
      "average": ((data["average"] * effective_count) + (sample * CSC_NUDGE_WEIGHT)) / total,
      "count": data["count"] + CSC_NUDGE_WEIGHT,
    }

    self.nudge_applied = True
    self.rebuild_lat_accel_curve()
    self.data_dirty = True
    self.flush_data()

  def rebuild_lat_accel_curve(self):
    grid_k = np.array([float(key) for key in self.required_curvatures])
    prior = np.interp(grid_k, PRIOR_CURVATURE_BP, PRIOR_LAT_ACCEL_V)

    blended = prior.copy()
    counts = np.zeros(len(grid_k))
    for i, key in enumerate(self.required_curvatures):
      data = self.curvature_data.get(key)
      if data:
        confidence = data["count"] / (data["count"] + CSC_PRIOR_COUNT)
        blended[i] = confidence * data["average"] + (1.0 - confidence) * prior[i]
        counts[i] = data["count"]

    blended = np.clip(blended, CSC_LAT_ACCEL_MIN, CSC_LAT_ACCEL_MAX)
    # Sparse buckets can invert the trend; comfort must not decrease with sharpness
    blended = np.maximum.accumulate(blended)

    self._curve_k = grid_k
    self._curve_a = blended

    if counts.sum() > 0:
      self.lateral_acceleration = float(np.average(blended, weights=counts))
    else:
      self.lateral_acceleration = DEFAULT_LATERAL_ACCELERATION

  def lat_accel_for_curvature(self, curvature):
    lat_accel = np.interp(np.abs(curvature), self._curve_k, self._curve_a)

    weather = self.starpilot_planner.starpilot_weather
    if weather.weather_id != 0:
      lat_accel = lat_accel * (1.0 - weather.reduce_lateral_acceleration)

    return lat_accel

  def reset(self, v_cruise):
    self.target = float(v_cruise)
    self.target_filter.x = float(v_cruise)
    self.target_filter.initialized = True
    self.seed_pending = True

  def update_target(self, v_ego, v_cruise):
    if not self.target_filter.initialized:
      self.reset(v_cruise)

    curvatures, distances = self.starpilot_planner.curve_profile
    if len(curvatures) == 0:
      raw_target = float(v_cruise)
    else:
      lat_accel = self.lat_accel_for_curvature(curvatures)
      point_speeds = np.sqrt(lat_accel / np.maximum(curvatures, 1e-4))
      point_speeds = np.maximum(point_speeds, CSC_MIN_SPEED)
      allowed_speeds = np.sqrt(point_speeds**2 + 2.0 * CSC_APPROACH_DECEL * np.maximum(distances, 0.0))
      raw_target = min(float(np.min(allowed_speeds)), float(v_cruise))

    # A fresh activation must not inherit a stale-high target and spend seconds
    # ramping down toward a curve the envelope already sees (e.g. engaging or
    # launching into a turn) — start at the envelope, floored at ego speed.
    if self.seed_pending:
      seed = min(float(v_cruise), max(raw_target, v_ego + CSC_EGO_HEADROOM))
      self.target = seed
      self.target_filter.x = seed
      self.seed_pending = False

    filtered = self.target_filter.update(raw_target)
    self.target = float(np.clip(filtered,
                                self.target - CSC_TARGET_DOWN_RATE * DT_MDL,
                                self.target + CSC_TARGET_UP_RATE * DT_MDL))
    self.target = max(self.target, min(raw_target, v_ego + CSC_EGO_HEADROOM))

    if self.target < v_cruise - CSC_ACTIVE_ON_DELTA:
      self.training_quiet_timer = CSC_TRAINING_QUIET_TIME
