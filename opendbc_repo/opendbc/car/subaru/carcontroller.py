import copy
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg, structs
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.lateral import apply_driver_steer_torque_limits, apply_std_steer_angle_limits, apply_steer_angle_limits_vm, common_fault_avoidance
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.avh import AvhStartup
from opendbc.car.subaru.values import CAR, DBC, GLOBAL_ES_ADDR, SUBARU_STOP_START_CARS, CanBus, CarControllerParams, SubaruFlags
from opendbc.car.vehicle_model import VehicleModel

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut

_SNG_ACC_MIN_DIST = 3
_SNG_ACC_MAX_DIST = 4.5
_LEGACY_2025_MADS_MIN_SPEED = 0.44704
_LEGACY_2025_MADS_MAX_STEER_ANGLE = 200.0
_ANGLE_OVERRIDE_CONFIRM_FRAMES = 2
_ANGLE_REENGAGE_MAX_STEER_RATE = 2.0
_LKA_REENGAGE_MAX_STEER_RATE = 15.0
_LKA_OVERRIDE_TORQUE_HIGH = 120
_LKA_OVERRIDE_TORQUE_LOW = 80
_ANGLE_MADS_MIN_SPEED = 0.44704
_ANGLE_MADS_MAX_STEER_ANGLE = 120.0
_ANGLE_STALL_ERROR_DEG = 5.0
_ANGLE_STALL_FRAMES = 8
_ASCENT_AOL_ARM_FRAMES = 30
_STOP_START_STARTUP_DELAY_FRAMES = 100
# StarPilot's first populated toggle message can arrive several seconds after
# the car controller starts while fingerprinting and settings settle.
_STOP_START_STARTUP_DEADLINE_FRAMES = 1000
_STOP_START_PULSE_FRAMES = 30
_STOP_START_PULSE_PERIOD_FRAMES = 5
_REDNECK_BUTTON_INTERVAL_FRAMES = 10
_REDNECK_BUTTON_COPIES = 2


def get_safety_CP():
  from opendbc.car.subaru.interface import CarInterface
  return CarInterface.get_non_essential_params("SUBARU_ASCENT")


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_torque_last = 0
    self.apply_steer_last = 0
    self.driver_override = False
    self.angle_override_confirm_frames = 0
    self.angle_lkas_active = False
    self.angle_handoff_active = False
    self.angle_stall_frames = 0
    self.angle_initialized = False
    self.ascent_angle_initialized = False
    self.ascent_aol_arm_frames = 0
    self.ascent_es_distance_counter_last = None
    self.es_distance_counter_last = None
    self.ascent_lkas_alert_counter_last = None
    self.lat_active_seen = False
    self.lka_armed = False
    self.lka_nudge_active = False
    self.in_drive_and_cruise_available = True
    self.standstill_in_drive = False
    self.left_line_vis_debounced = None
    self.right_line_vis_debounced = None
    self.left_line_debounce_cnt = 0
    self.right_line_debounce_cnt = 0

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])
    self.main_bus = CanBus.main_for_cp(CP)
    self.angle_bus = CanBus.angle_for_cp(CP)
    self.status_bus = CanBus.camera if CP.flags & SubaruFlags.D_PLATFORM_CAMERA else CanBus.main

    if CP.flags & SubaruFlags.LKAS_ANGLE and CP.carFingerprint != CAR.SUBARU_OUTBACK_2023:
      self.VM = VehicleModel(get_safety_CP())

    if CP.flags & SubaruFlags.LKAS_ANGLE:
      self.angle_filter = FirstOrderFilter(0.0, 0.3, DT_CTRL)

    self.prev_close_distance = 0
    self.epb_resume_frames_remaining = -1
    self.last_standstill_frame = 0
    self.stop_start_attempted = False
    self.stop_start_request_started = False
    self.stop_start_request_frame = 0
    self.stop_start_initial_state = None
    self.stop_start_counter = 0
    self.stop_start_acknowledged = False
    self.last_redneck_button_frame = 0
    self.avh_startup = AvhStartup()

  def _stop_start_off_request(self, CC, CS, starpilot_toggles):
    """Send one bounded Subaru Stop/Start OFF request after ignition.

    This is intentionally opt-in and limited to a stationary vehicle in
    Park/Neutral. A single ignition session gets at most one attempt.
    """
    if self.CP.carFingerprint not in SUBARU_STOP_START_CARS or \
       not getattr(starpilot_toggles, "subaru_stop_start_off", False) or self.stop_start_attempted:
      return None

    if self.frame > _STOP_START_STARTUP_DEADLINE_FRAMES or getattr(CC, "enabled", False):
      self.stop_start_attempted = True
      return None

    if self.frame < _STOP_START_STARTUP_DELAY_FRAMES or not getattr(getattr(CS, "out", None), "canValid", True):
      return None

    out = CS.out
    if not getattr(out, "standstill", False) or out.gearShifter not in (
      structs.CarState.GearShifter.park,
      structs.CarState.GearShifter.neutral,
    ):
      return None

    dashlights_msg = getattr(CS, "dashlights_msg", None)
    if not dashlights_msg:
      return None

    if not self.stop_start_request_started:
      self.stop_start_request_started = True
      self.stop_start_request_frame = self.frame
      self.stop_start_initial_state = getattr(CS, "stop_start_state", None)
      self.stop_start_counter = (int(dashlights_msg.get("COUNTER", 0)) + 1) % 0x10

    current_state = getattr(CS, "stop_start_state", None)
    if self.stop_start_initial_state is not None and self.stop_start_initial_state != 3 and current_state == 3:
      self.stop_start_attempted = True
      self.stop_start_acknowledged = True
      return None

    elapsed = self.frame - self.stop_start_request_frame
    if elapsed >= _STOP_START_PULSE_FRAMES:
      self.stop_start_attempted = True
      return None

    if elapsed % _STOP_START_PULSE_PERIOD_FRAMES != 0:
      return None

    msg = subarucan.create_stop_start_control(
      self.packer, dashlights_msg, raw_dat=getattr(CS, "dashlights_dat", None),
      counter=self.stop_start_counter, bus=CanBus.alt_for_cp(self.CP),
    )
    self.stop_start_counter = (self.stop_start_counter + 1) % 0x10
    return msg

  def _reset_angle_handoff(self):
    self.driver_override = False
    self.angle_override_confirm_frames = 0
    self.angle_handoff_active = False
    self.angle_stall_frames = 0

  def _angle_manual_handoff(self, CS, lat_active):
    if not lat_active:
      self._reset_angle_handoff()
      return False

    driver_override = self._update_angle_driver_override(CS)
    steering_rate = abs(getattr(CS.out, "steeringRateDeg", 0.0))
    reengage_max_rate = _LKA_REENGAGE_MAX_STEER_RATE if self.lka_nudge_active else _ANGLE_REENGAGE_MAX_STEER_RATE
    if driver_override:
      self.angle_handoff_active = True
      return True

    if self.angle_handoff_active:
      if steering_rate > reengage_max_rate:
        return True

      self.angle_handoff_active = False
      return True

    if not self.angle_lkas_active and steering_rate > reengage_max_rate:
      self.angle_handoff_active = True
      return True

    # Stall guard: if commanded angle and actual wheel angle diverge significantly while
    # LKAS is active, the driver is holding the wheel against openpilot.
    # Yield control before Subaru's EPS rack detects a motor stall and trips Steer_Error_1.
    angle_err = abs(self.apply_steer_last - getattr(CS.out, "steeringAngleDeg", 0.0))
    if self.angle_lkas_active and angle_err > _ANGLE_STALL_ERROR_DEG:
      self.angle_stall_frames += 1
      if self.angle_stall_frames >= _ANGLE_STALL_FRAMES:
        self.angle_handoff_active = True
        return True
    else:
      self.angle_stall_frames = 0

    return False

  def _update_angle_driver_override(self, CS):
    """Debounce the higher-confidence raw torque override signal for angle cars."""
    abs_torque = abs(getattr(CS.out, "steeringTorque", 0.0))
    torque_low = min(self.p.STEER_OVERRIDE_TORQUE_LOW, _LKA_OVERRIDE_TORQUE_LOW) if self.lka_nudge_active else self.p.STEER_OVERRIDE_TORQUE_LOW
    torque_high = min(self.p.STEER_OVERRIDE_TORQUE_HIGH, _LKA_OVERRIDE_TORQUE_HIGH) if self.lka_nudge_active else self.p.STEER_OVERRIDE_TORQUE_HIGH
    if self.driver_override:
      if abs_torque < torque_low:
        self.driver_override = False
    elif abs_torque > torque_high:
      self.angle_override_confirm_frames += 1
      if self.angle_override_confirm_frames >= _ANGLE_OVERRIDE_CONFIRM_FRAMES:
        self.driver_override = True
        self.angle_override_confirm_frames = 0
    else:
      self.angle_override_confirm_frames = 0

    return self.driver_override

  def _ascent_aol_ready(self, ready):
    if not ready:
      self.ascent_aol_arm_frames = 0
      return False

    self.ascent_aol_arm_frames = min(self.ascent_aol_arm_frames + 1, _ASCENT_AOL_ARM_FRAMES)
    return self.ascent_aol_arm_frames >= _ASCENT_AOL_ARM_FRAMES

  def lateral_angle(self, CC, CS, apply_angle=None):
    if not self.angle_initialized:
      self.apply_steer_last = CS.out.steeringAngleDeg
      self.angle_initialized = True
      self.ascent_angle_initialized = True

    if apply_angle is None:
      apply_angle = CC.actuators.steeringAngleDeg

    self.lka_armed = bool(getattr(CS, "lka_armed", False))
    self.lka_nudge_active = bool(getattr(CS, "lka_nudge_active", False))

    if self.CP.carFingerprint == CAR.SUBARU_LEGACY_2025:
      mads_only = CC.latActive and not CC.enabled
      if mads_only:
        apply_angle = float(np.clip(apply_angle, -_LEGACY_2025_MADS_MAX_STEER_ANGLE, _LEGACY_2025_MADS_MAX_STEER_ANGLE))

      mads_only_ok = CS.out.vEgoRaw > _LEGACY_2025_MADS_MIN_SPEED and \
        abs(CS.out.steeringAngleDeg) <= (_LEGACY_2025_MADS_MAX_STEER_ANGLE + 10.0)
      lkas_available = CC.latActive and (not mads_only or mads_only_ok) and \
        CS.out.gearShifter == structs.CarState.GearShifter.drive and not CS.out.standstill

      manual_handoff = self._angle_manual_handoff(CS, lkas_available)
      lkas_active = lkas_available and not manual_handoff

      apply_steer = apply_std_steer_angle_limits(
        apply_angle,
        self.apply_steer_last,
        CS.out.vEgoRaw,
        CS.out.steeringAngleDeg,
        lkas_active,
        self.p.LEGACY_2025_ANGLE_LIMITS,
      )
      self.apply_steer_last = apply_steer
      self.angle_lkas_active = lkas_active
      return subarucan.create_steering_control_angle(self.packer, apply_steer, lkas_active, self.angle_bus)

    if self.CP.carFingerprint in (CAR.SUBARU_ASCENT_2023, CAR.SUBARU_OUTBACK_2023):
      mads_only = CC.latActive and not CC.enabled
      mads_only_ok = CS.out.vEgoRaw > _ANGLE_MADS_MIN_SPEED and \
        abs(CS.out.steeringAngleDeg) < _ANGLE_MADS_MAX_STEER_ANGLE
      lkas_available = CC.latActive and (not mads_only or mads_only_ok) and \
        CS.out.gearShifter == structs.CarState.GearShifter.drive and not CS.out.standstill
      if self.CP.carFingerprint == CAR.SUBARU_ASCENT_2023:
        if mads_only:
          cruise_available = getattr(getattr(CS.out, "cruiseState", None), "available", True)
          lkas_available = self._ascent_aol_ready(lkas_available and cruise_available)
        else:
          self.ascent_aol_arm_frames = _ASCENT_AOL_ARM_FRAMES if lkas_available else 0

      if self.CP.carFingerprint == CAR.SUBARU_OUTBACK_2023:
        manual_handoff = not self.angle_lkas_active and \
          abs(getattr(CS.out, "steeringRateDeg", 0.0)) > _ANGLE_REENGAGE_MAX_STEER_RATE
      else:
        manual_handoff = self._angle_manual_handoff(CS, lkas_available)
      lkas_active = lkas_available and not manual_handoff

      apply_steer = apply_std_steer_angle_limits(
        apply_angle,
        self.apply_steer_last,
        CS.out.vEgoRaw,
        CS.out.steeringAngleDeg,
        lkas_active,
        self.p.FIXED_ANGLE_LIMITS,
      )
      self.apply_steer_last = apply_steer
      self.angle_lkas_active = lkas_active
      return subarucan.create_steering_control_angle(self.packer, apply_steer, lkas_active, self.angle_bus)

    mads_only = CC.latActive and not getattr(CC, "enabled", False)
    mads_only_ok = CS.out.vEgoRaw > _ANGLE_MADS_MIN_SPEED and \
      abs(CS.out.steeringAngleDeg) < _ANGLE_MADS_MAX_STEER_ANGLE
    lkas_available = CC.latActive and (not mads_only or mads_only_ok) and \
      getattr(CS.out, "gearShifter", structs.CarState.GearShifter.drive) == structs.CarState.GearShifter.drive and \
      not getattr(CS.out, "standstill", False)
    manual_handoff = self._angle_manual_handoff(CS, lkas_available)
    lat_active = lkas_available and not self.driver_override and not manual_handoff
    if lat_active and not self.angle_lkas_active:
      self.apply_steer_last = CS.out.steeringAngleDeg
    apply_steer = apply_steer_angle_limits_vm(
      apply_angle,
      self.apply_steer_last,
      CS.out.vEgoRaw,
      CS.out.steeringAngleDeg,
      lat_active,
      self.p,
      self.VM,
    )

    if not lat_active:
      apply_steer = CS.out.steeringAngleDeg

    self.apply_steer_last = apply_steer
    self.angle_lkas_active = lat_active
    return subarucan.create_steering_control_angle(self.packer, apply_steer, lat_active, self.angle_bus)

  def lateral_torque(self, CC, CS):
    apply_torque = int(round(CC.actuators.torque * self.p.STEER_MAX))
    apply_torque = apply_driver_steer_torque_limits(apply_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

    if not CC.latActive:
      apply_torque = 0

    self.apply_torque_last = apply_torque

    if self.CP.flags & SubaruFlags.PREGLOBAL:
      return subarucan.create_preglobal_steering_control(
        self.packer, self.frame // self.p.STEER_STEP, apply_torque, CC.latActive,
      )

    apply_steer_req = CC.latActive
    if self.CP.flags & SubaruFlags.STEER_RATE_LIMITED:
      self.steer_rate_counter, apply_steer_req = common_fault_avoidance(
        abs(CS.out.steeringRateDeg) > MAX_STEER_RATE,
        apply_steer_req,
        self.steer_rate_counter,
        MAX_STEER_RATE_FRAMES,
      )

    return subarucan.create_steering_control(self.packer, apply_torque, apply_steer_req)

  def _lkas_status_active(self, CC):
    if self.lka_nudge_active or self.lka_armed or not self.in_drive_and_cruise_available:
      return False
    if self.CP.carFingerprint in (CAR.SUBARU_ASCENT_2023, CAR.SUBARU_OUTBACK_2023, CAR.SUBARU_LEGACY_2025):
      if self.angle_lkas_active:
        return True
      return bool(CC.latActive and self.standstill_in_drive and not self.driver_override and not self.angle_handoff_active)
    return CC.latActive

  def _ascent_lkas_alert(self, CC, CS):
    if self.CP.carFingerprint not in (CAR.SUBARU_ASCENT_2023, CAR.SUBARU_LEGACY_2025) or not self.CP.flags & SubaruFlags.ASCENT_LKAS_ALERT:
      return None
    dat = getattr(CS, "es_lkas_alert_dat", b"")
    cruise_state = getattr(getattr(CS, "out", None), "cruiseState", None)
    armed = bool(cruise_state.available) if cruise_state is not None else False
    msg = subarucan.create_ascent_lkas_alert(dat, self.angle_lkas_active, CC.hudControl.visualAlert, armed=armed)
    if msg is None or (dat[1] & 0xf) == self.ascent_lkas_alert_counter_last:
      return None
    self.ascent_lkas_alert_counter_last = dat[1] & 0xf
    return msg

  def update(self, CC, CS, now_nanos, starpilot_toggles=None):
    self.lka_armed = bool(getattr(CS, "lka_armed", False))
    self.lka_nudge_active = bool(getattr(CS, "lka_nudge_active", False))
    cs_out = getattr(CS, "out", None)
    in_drive = getattr(cs_out, "gearShifter", None) == structs.CarState.GearShifter.drive
    cruise_state = getattr(cs_out, "cruiseState", None)
    cruise_available = bool(cruise_state.available) if cruise_state is not None else bool(
      CC.enabled or CC.latActive or self.lka_armed or self.lka_nudge_active
    )
    self.in_drive_and_cruise_available = bool(in_drive and cruise_available)
    self.standstill_in_drive = bool(
      in_drive and (getattr(cs_out, "standstill", False) or getattr(cs_out, "vEgoRaw", 0.0) < _LEGACY_2025_MADS_MIN_SPEED)
    )
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel
    subaru_redneck_cruise = bool(
      self.CP.carFingerprint == CAR.SUBARU_IMPREZA_2020 and
      getattr(starpilot_toggles, "subaru_redneck_cruise", False)
    )

    can_sends = []

    stop_start_msg = self._stop_start_off_request(CC, CS, starpilot_toggles)
    if stop_start_msg is not None:
      can_sends.append(stop_start_msg)

    if self.CP.carFingerprint == CAR.SUBARU_LEGACY_2025:
      can_sends.extend(self.avh_startup.update(
        now_nanos / 1e9, getattr(CS, "avh_frames", {}),
        getattr(starpilot_toggles, "subaru_avh_on", False), getattr(CS.out, "canValid", False),
        CC.enabled or CC.latActive or CC.longActive,
      ))

    # *** steering ***
    if self.CP.flags & SubaruFlags.LKAS_ANGLE:
      # Filter at the control-loop rate, including frames without a steering CAN message.
      v_ego = getattr(cs_out, "vEgo", 0.0)
      v_ego_raw = getattr(cs_out, "vEgoRaw", v_ego)
      steer_angle = getattr(cs_out, "steeringAngleDeg", actuators.steeringAngleDeg)
      standstill = bool(getattr(cs_out, "standstill", False))
      mads_only = bool(CC.latActive and not getattr(CC, "enabled", False))
      if self.CP.carFingerprint == CAR.SUBARU_LEGACY_2025:
        mads_ready = v_ego_raw > _LEGACY_2025_MADS_MIN_SPEED and abs(steer_angle) <= (_LEGACY_2025_MADS_MAX_STEER_ANGLE + 10.0)
      else:
        mads_ready = v_ego_raw > _ANGLE_MADS_MIN_SPEED and abs(steer_angle) < _ANGLE_MADS_MAX_STEER_ANGLE
      angle_rack_ready = bool(in_drive and not standstill and (not mads_only or mads_ready))

      if not CC.latActive:
        self.angle_filter.x = actuators.steeringAngleDeg
      elif not angle_rack_ready or self.angle_handoff_active or self.driver_override:
        self.angle_filter.x = steer_angle
      elif self.lka_nudge_active:
        # LaneKeepController already ramps curvature smoothly; avoid double-filtering LKA nudges.
        self.angle_filter.x = actuators.steeringAngleDeg
      else:
        self.angle_filter.update_alpha(float(np.interp(v_ego, [5.0, 10.0, 20.0], [0.3, 0.1, 0.0])))
        self.angle_filter.update(actuators.steeringAngleDeg)

    if (self.frame % self.p.STEER_STEP) == 0:
      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        can_sends.append(self.lateral_angle(CC, CS, apply_angle=self.angle_filter.x))
      else:
        can_sends.append(self.lateral_torque(CC, CS))

    # *** stop and go ***
    subaru_sng_manual_parking_brake = getattr(starpilot_toggles, "subaru_sng_manual_parking_brake", False)
    if getattr(starpilot_toggles, "subaru_sng", False):
      throttle_cmd, speed_cmd = self.stop_and_go(CC, CS, subaru_sng_manual_parking_brake)

    # *** longitudinal ***

    if CC.longActive:
      apply_throttle = int(round(np.interp(actuators.accel, CarControllerParams.THROTTLE_LOOKUP_BP, CarControllerParams.THROTTLE_LOOKUP_V)))
      apply_rpm = int(round(np.interp(actuators.accel, CarControllerParams.RPM_LOOKUP_BP, CarControllerParams.RPM_LOOKUP_V)))
      apply_brake = int(round(np.interp(actuators.accel, CarControllerParams.BRAKE_LOOKUP_BP, CarControllerParams.BRAKE_LOOKUP_V)))

      # limit min and max values
      cruise_throttle = np.clip(apply_throttle, CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
      cruise_rpm = np.clip(apply_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
      cruise_brake = np.clip(apply_brake, CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
    else:
      cruise_throttle = CarControllerParams.THROTTLE_INACTIVE
      cruise_rpm = CarControllerParams.RPM_MIN
      cruise_brake = CarControllerParams.BRAKE_MIN

    # *** alerts and pcm cancel ***
    lkas_alert_msg = self._ascent_lkas_alert(CC, CS)
    if lkas_alert_msg is not None:
      can_sends.append(lkas_alert_msg)

    if self.CP.flags & SubaruFlags.PREGLOBAL:
      if self.frame % 5 == 0:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))

      if getattr(starpilot_toggles, "subaru_sng", False):
        can_sends.append(subarucan.create_preglobal_throttle(self.packer, CS.throttle_msg["COUNTER"] + 1, CS.throttle_msg,
                                                             throttle_cmd))
        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_preglobal_brake_pedal(self.packer, CS.brake_pedal_msg,
                                                                  speed_cmd))
    else:
      if self.frame % 10 == 0:
        if getattr(CS, "es_dashstatus_msg", None) is not None:
          can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, CC.enabled,
                                                          self.CP.openpilotLongitudinalControl and not subaru_redneck_cruise,
                                                          CC.longActive, hud_control.leadVisible,
                                                          self.status_bus))

        if getattr(CS, "es_lkas_state_msg", None) is not None:
          if not self.in_drive_and_cruise_available:
            # Cruise Main OFF (Cruise_On == 0) or shifted out of Drive:
            # Hide BOTH the steering wheel icon (0) and lane lines (0) completely.
            self.lat_active_seen = False
            self.left_line_vis_debounced = None
            self.right_line_vis_debounced = None
            self.left_line_debounce_cnt = 0
            self.right_line_debounce_cnt = 0
            wheel_ready = False
            left_line = 0
            right_line = 0
          else:
            if CC.latActive and not self.lka_nudge_active:
              self.lat_active_seen = True

            # Steering wheel icon states on cluster (LKAS_Dash_State) when Cruise_On == 1 in Drive:
            # 2 (Green): Continuous Centering actively steering (_lkas_status_active(CC) == True)
            # 1 (White): LKA-Only standby/nudge, OR Continuous Centering paused at standstill / manually overridden
            wheel_ready = True

            # Line debouncing (immediate seed when freshly armed; 1.0s = 10 frames @ 10 Hz debounce thereafter)
            raw_left_vis = bool(hud_control.leftLaneVisible)
            raw_right_vis = bool(hud_control.rightLaneVisible)

            if self.left_line_vis_debounced is None:
              self.left_line_vis_debounced = raw_left_vis
              self.left_line_debounce_cnt = 0
            elif raw_left_vis != self.left_line_vis_debounced:
              self.left_line_debounce_cnt += 1
              if self.left_line_debounce_cnt >= 10:
                self.left_line_vis_debounced = raw_left_vis
                self.left_line_debounce_cnt = 0
            else:
              self.left_line_debounce_cnt = 0

            if self.right_line_vis_debounced is None:
              self.right_line_vis_debounced = raw_right_vis
              self.right_line_debounce_cnt = 0
            elif raw_right_vis != self.right_line_vis_debounced:
              self.right_line_debounce_cnt += 1
              if self.right_line_debounce_cnt >= 10:
                self.right_line_vis_debounced = raw_right_vis
                self.right_line_debounce_cnt = 0
            else:
              self.right_line_debounce_cnt = 0

            steer_angle_deg = abs(getattr(CS.out, "steeringAngleDeg", 0.0))
            # In Continuous Centering (not self.lka_armed), do NOT cut out lines on 45-deg wheel angle or speed;
            # track all the way down to 0 mph. Cornering gate (>= 45 deg) applies only in LKA-Only mode.
            steer_gate = (steer_angle_deg >= 45.0) if self.lka_armed else False
            left_blinker = bool(getattr(CS.out, "leftBlinker", False))
            right_blinker = bool(getattr(CS.out, "rightBlinker", False))

            if hud_control.leftLaneDepart:
              left_line = 3  # Amber nudge line on cluster (immediate, no debounce)
            elif steer_gate or left_blinker:
              left_line = 0
            else:
              left_line = 1 if self.left_line_vis_debounced else 0

            if hud_control.rightLaneDepart:
              right_line = 3  # Amber nudge line on cluster (immediate, no debounce)
            elif steer_gate or right_blinker:
              right_line = 0
            else:
              right_line = 1 if self.right_line_vis_debounced else 0

          can_sends.append(subarucan.create_es_lkas_state(
            self.packer, self.frame // 10, CS.es_lkas_state_msg, self._lkas_status_active(CC), hud_control.visualAlert,
            left_line, right_line,
            hud_control.leftLaneDepart, hud_control.rightLaneDepart, self.status_bus,
            ready=wheel_ready,
            in_drive=in_drive,
          ))

        if (self.CP.flags & SubaruFlags.SEND_INFOTAINMENT) and getattr(CS, "es_infotainment_msg", None) is not None:
          can_sends.append(subarucan.create_es_infotainment(self.packer, self.frame // 10, CS.es_infotainment_msg,
                                                           hud_control.visualAlert, self.status_bus))

      if getattr(starpilot_toggles, "subaru_sng", False):
        can_sends.append(subarucan.create_throttle(self.packer, CS.throttle_msg["COUNTER"] + 1, CS.throttle_msg,
                                                   throttle_cmd))
        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_brake_pedal(self.packer, self.frame // 2, CS.brake_pedal_msg,
                                                        speed_cmd, pcm_cancel_cmd))

      if self.CP.openpilotLongitudinalControl and not subaru_redneck_cruise:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, self.frame // 5, CS.es_status_msg,
                                                      self.CP.openpilotLongitudinalControl, CC.longActive, cruise_rpm))

          can_sends.append(subarucan.create_es_brake(self.packer, self.frame // 5, CS.es_brake_msg,
                                                     self.CP.openpilotLongitudinalControl, CC.longActive, cruise_brake))

          can_sends.append(subarucan.create_es_distance(self.packer, self.frame // 5, CS.es_distance_msg, 0, pcm_cancel_cmd,
                                                        self.CP.openpilotLongitudinalControl, cruise_brake > 0, cruise_throttle))
      else:
        cancel_frame_ready = True
        if (self.CP.flags & SubaruFlags.GLOBAL_GEN2) or self.CP.carFingerprint == CAR.SUBARU_ASCENT_2023:
          es_distance_msg = getattr(CS, "es_distance_msg", None)
          if es_distance_msg is not None:
            stock_counter = es_distance_msg["COUNTER"]
            cancel_frame_ready = stock_counter != self.es_distance_counter_last
            self.es_distance_counter_last = stock_counter
            self.ascent_es_distance_counter_last = stock_counter
        if pcm_cancel_cmd and cancel_frame_ready:
          if not (self.CP.flags & SubaruFlags.HYBRID):
            bus = CanBus.alt_for_cp(self.CP) if self.CP.flags & SubaruFlags.GLOBAL_GEN2 else self.main_bus
            can_sends.append(subarucan.create_es_distance(self.packer, (CS.es_distance_msg["COUNTER"] + 1) % 0x10, CS.es_distance_msg, bus, pcm_cancel_cmd))

      if subaru_redneck_cruise:
        redneck_button = {
          1: subarucan.CRUISE_BUTTON_RESUME,
          2: subarucan.CRUISE_BUTTON_SET,
        }.get(getattr(CS, "redneck_send_button", 0))
        cruise_buttons_msg = getattr(CS, "cruise_buttons_msg", None)
        if redneck_button and cruise_buttons_msg and self.frame - self.last_redneck_button_frame >= _REDNECK_BUTTON_INTERVAL_FRAMES:
          counter = (int(cruise_buttons_msg["COUNTER"]) + 1) % 0x10
          for copy_idx in range(_REDNECK_BUTTON_COPIES):
            can_sends.append(subarucan.create_cruise_buttons(
              self.packer, counter + copy_idx, cruise_buttons_msg, redneck_button, self.main_bus,
            ))
          self.last_redneck_button_frame = self.frame

      if self.CP.flags & SubaruFlags.DISABLE_EYESIGHT:
        # Tester present (keeps eyesight disabled)
        if self.frame % 100 == 0:
          can_sends.append(make_tester_present_msg(GLOBAL_ES_ADDR, CanBus.camera, suppress_response=True))

        # Create all of the other eyesight messages to keep the rest of the car happy when eyesight is disabled
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_highbeamassist(self.packer))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_static_1(self.packer))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_es_static_2(self.packer))

    new_actuators = actuators.as_builder() if hasattr(actuators, "as_builder") else copy.copy(actuators)
    if self.CP.flags & SubaruFlags.LKAS_ANGLE:
      new_actuators.steeringAngleDeg = self.apply_steer_last
    else:
      new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
      new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends

  def stop_and_go(self, CC, CS, manual_parking_brake=False):
    throttle_cmd = False
    speed_cmd = False

    if not CC.enabled or not CC.hudControl.leadVisible:
      return throttle_cmd, speed_cmd

    close_distance = CS.close_distance
    if not CS.out.standstill:
      self.last_standstill_frame = self.frame

    standstill_timers = (0.75, 0.8) if self.CP.flags & SubaruFlags.PREGLOBAL else (0.5, 0.55)
    standstill_duration = (self.frame - self.last_standstill_frame) * DT_CTRL
    in_standstill_hold = standstill_duration > standstill_timers[0]
    if standstill_duration >= standstill_timers[1]:
      self.last_standstill_frame = self.frame

    if manual_parking_brake or not (self.CP.flags & SubaruFlags.PREGLOBAL):
      speed_cmd = in_standstill_hold

    should_resume = (
      CS.out.standstill and
      _SNG_ACC_MIN_DIST < close_distance < _SNG_ACC_MAX_DIST and
      close_distance > self.prev_close_distance
    )
    if should_resume:
      self.epb_resume_frames_remaining = 15

    throttle_cmd = self.epb_resume_frames_remaining > 0
    if self.epb_resume_frames_remaining > 0:
      self.epb_resume_frames_remaining -= 1

    self.prev_close_distance = close_distance
    return throttle_cmd, speed_cmd
