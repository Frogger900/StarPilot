import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.carcontroller import CarController
from opendbc.car.subaru.carstate import CarState
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.tests.test_ascent_angle import setup_controller, step
from opendbc.car.subaru.values import CAR, CanBus, DBC, FROGGER_ANGLE_SIBLINGS, SUBARU_LKA_CARS
from opendbc.safety.tests import test_subaru_gen2_angle_pair as pair


def _step_to_hud(controller, command, state):
  while controller.frame % 10 != 0:
    controller.update(command.as_reader(), state, 1_000_000_000 + controller.frame * 10_000_000)
  _, sends = controller.update(command.as_reader(), state, 1_000_000_000 + controller.frame * 10_000_000)
  hud = next(frame for frame in sends if frame[0] == 0x322)
  parser = CANParser(DBC[controller.CP.carFingerprint][Bus.pt], [('ES_LKAS_State', 0)], hud[2])
  parser.update([(1, [hud])])
  return parser.vl['ES_LKAS_State']


class TestSubaruLka(unittest.TestCase):
  def test_frogger_angle_siblings_stay_excluded_from_subaru_lka_cars(self):
    self.assertEqual(SUBARU_LKA_CARS, frozenset((CAR.SUBARU_LEGACY_2025,)))
    for sibling in FROGGER_ANGLE_SIBLINGS:
      self.assertNotIn(sibling, SUBARU_LKA_CARS)
      _, controller, command, state = setup_controller(sibling)
      state.lka_enabled = True
      state.lka_armed = True
      command.latActive = True
      _, decoded = step(controller, command, state)
      self.assertEqual(decoded['SET_3'], 3, sibling)
  def test_lkas_dash_state_toggle_emits_button_event_and_arms_lka(self):
    cp = CarInterface.get_params(CAR.SUBARU_LEGACY_2025, {0: {}, 1: {}, 2: {}}, [], False, False, False)
    _, frames = pair.TestSubaruGen2AnglePair.sources(CAR.SUBARU_LEGACY_2025)
    packer = CANPacker(DBC[CAR.SUBARU_LEGACY_2025][Bus.pt])
    state = CarState(cp)
    self.assertFalse(state.lka_enabled)
    state.lka_enabled = True
    parsers = state.get_can_parsers(cp)
    state.update(parsers)
    for parser in parsers.values():
      parser.update([(1_000_000_000, frames)])
    ret = state.update(parsers)
    self.assertTrue(ret.cruiseState.available)
    self.assertFalse(state.lka_armed)
    self.assertEqual(len(ret.buttonEvents), 0)

    # Flip LKAS_Dash_State (0 -> 1) on bus 2: arms LKA once and holds Type.lkas press/release for 5 frames.
    lkas_on = packer.make_can_msg('ES_LKAS_State', CanBus.camera, {'LKAS_Dash_State': 1})
    parsers[Bus.cam].update([(1_020_000_000, [lkas_on])])
    ret = state.update(parsers)
    self.assertTrue(state.lka_armed)
    lkas_events = [be for be in ret.buttonEvents if be.type == structs.CarState.ButtonEvent.Type.lkas]
    self.assertEqual([be.pressed for be in lkas_events], [True, False])
    for idx in range(4):
      ret = state.update(parsers)
      self.assertTrue(state.lka_armed)
      lkas_events = [be for be in ret.buttonEvents if be.type == structs.CarState.ButtonEvent.Type.lkas]
      self.assertEqual([be.pressed for be in lkas_events], [True, False], f"hold frame {idx + 2}")
    ret = state.update(parsers)
    self.assertTrue(state.lka_armed)
    self.assertEqual([be for be in ret.buttonEvents if be.type == structs.CarState.ButtonEvent.Type.lkas], [])

    # Flip LKAS_Dash_State (1 -> 0): disarms LKA cleanly.
    lkas_off = packer.make_can_msg('ES_LKAS_State', CanBus.camera, {'LKAS_Dash_State': 0})
    parsers[Bus.cam].update([(1_040_000_000, [lkas_off])])
    ret = state.update(parsers)
    self.assertFalse(state.lka_armed)

    # Re-arm, then drop Cruise_On -> lka_armed resets to False.
    parsers[Bus.cam].update([(1_060_000_000, [lkas_on])])
    state.update(parsers)
    self.assertTrue(state.lka_armed)
    dash_off = packer.make_can_msg('ES_DashStatus', CanBus.camera, {'Cruise_On': 0})
    parsers[Bus.cam].update([(1_080_000_000, [dash_off])])
    state.update(parsers)
    self.assertFalse(state.lka_armed)

  def test_carcontroller_lka_standby_and_nudge_hud_and_override(self):
    _, controller, command, state = setup_controller(CAR.SUBARU_LEGACY_2025)
    state.lka_armed = True
    command.enabled = False
    command.latActive = False
    command.hudControl.leftLaneVisible = True
    command.hudControl.rightLaneVisible = False
    state.out.steeringAngleDeg = 0.0
    state.out.steeringRateDeg = 0.0
    state.out.steeringTorque = 0.0

    # Standby: no steering torque/angle request, white wheel icon (1), immediate left line visibility (1).
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 1)
    self.assertEqual(hud['LKAS_Right_Line_Visible'], 0)

    # Cornering gate (< 20 deg keeps lines; >= 20 deg blanks standby lines) and blinker gate blank standby lines immediately in LKA mode.
    state.out.steeringAngleDeg = 19.0
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 1)
    state.out.steeringAngleDeg = 20.0
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 0)
    state.out.steeringAngleDeg = 0.0
    state.out.leftBlinker = True
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 0)
    state.out.leftBlinker = False

    # Active nudge: angle requested with SET_3=2 and smooth 0 deg/s onset ramp, wheel icon white (1), left line amber (3).
    command.latActive = True
    command.hudControl.leftLaneDepart = True
    command.actuators.steeringAngleDeg = -8.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    self.assertEqual(decoded['SET_3'], 2)
    self.assertLess(decoded['LKAS_Output'], 0.0)
    self.assertGreater(decoded['LKAS_Output'], -0.05)  # 0.05 ramp on first step avoids sudden onset jerk
    for _ in range(25):
      _, decoded = step(controller, command, state)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 3)

    # Opposing driver torque <= 175 stays active; > 175 overrides LKA nudge, then clears handoff and re-engages below 110 at <= 25 deg/s.
    state.out.steeringTorque = 150.0
    step(controller, command, state)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    state.out.steeringTorque = 180.0
    step(controller, command, state)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)
    state.out.steeringTorque = 105.0
    state.out.steeringRateDeg = 20.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)

    # Engaging EyeSight ACC (enabled=True) while in LKA mode must stay in LKA mode.
    command.enabled = True
    state.out.cruiseState.enabled = True
    command.latActive = False
    command.hudControl.leftLaneDepart = False
    state.out.steeringTorque = 0.0
    state.out.steeringRateDeg = 0.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)

    command.latActive = True
    command.hudControl.leftLaneDepart = True
    command.actuators.steeringAngleDeg = -8.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 3)

  def test_four_aol_lka_mode_combinations_and_lkas_button_transitions(self):
    cp = CarInterface.get_params(CAR.SUBARU_LEGACY_2025, {0: {}, 1: {}, 2: {}}, [], False, False, False)
    _, frames = pair.TestSubaruGen2AnglePair.sources(CAR.SUBARU_LEGACY_2025)
    packer = CANPacker(DBC[CAR.SUBARU_LEGACY_2025][Bus.pt])

    def init_state(aol_enabled: bool, lka_enabled: bool, last_centering: bool = True):
      state = CarState(cp)
      state.aol_enabled = aol_enabled
      state.lka_enabled = lka_enabled
      state.last_lateral_mode_centering = last_centering
      parsers = state.get_can_parsers(cp)
      state.update(parsers)
      for parser in parsers.values():
        parser.update([(1_000_000_000, frames)])
      state.update(parsers)
      return state, parsers

    def press_lkas(state, parsers, ts_ns: int, dash_state: int):
      msg = packer.make_can_msg('ES_LKAS_State', CanBus.camera, {'LKAS_Dash_State': dash_state})
      parsers[Bus.cam].update([(ts_ns, [msg])])
      return state.update(parsers)

    # 1. Both AOL + LKA ON: Centering <-> LKA, persisting last_lateral_mode_centering.
    state, parsers = init_state(True, True, last_centering=True)
    self.assertFalse(state.lka_armed)
    self.assertFalse(state.lateral_off)
    press_lkas(state, parsers, 1_020_000_000, 1)
    self.assertTrue(state.lka_armed)
    self.assertFalse(state.lateral_off)
    self.assertFalse(state.last_lateral_mode_centering)
    press_lkas(state, parsers, 1_040_000_000, 0)
    self.assertFalse(state.lka_armed)
    self.assertFalse(state.lateral_off)
    self.assertTrue(state.last_lateral_mode_centering)

    state_lka_boot, _ = init_state(True, True, last_centering=False)
    self.assertTrue(state_lka_boot.lka_armed)
    self.assertFalse(state_lka_boot.lateral_off)

    # 2. Only AOL ON: AOL <-> All Off.
    state, parsers = init_state(True, False)
    self.assertFalse(state.lka_armed)
    self.assertFalse(state.lateral_off)
    press_lkas(state, parsers, 1_020_000_000, 1)
    self.assertFalse(state.lka_armed)
    self.assertTrue(state.lateral_off)
    press_lkas(state, parsers, 1_040_000_000, 0)
    self.assertFalse(state.lka_armed)
    self.assertFalse(state.lateral_off)

    # 3. Only LKA ON: LKA <-> All Off.
    state, parsers = init_state(False, True)
    self.assertTrue(state.lka_armed)
    self.assertFalse(state.lateral_off)
    press_lkas(state, parsers, 1_020_000_000, 1)
    self.assertFalse(state.lka_armed)
    self.assertTrue(state.lateral_off)
    press_lkas(state, parsers, 1_040_000_000, 0)
    self.assertTrue(state.lka_armed)
    self.assertFalse(state.lateral_off)

    # 4. Both OFF: forced All Off even after LKAS presses.
    state, parsers = init_state(False, False)
    self.assertFalse(state.lka_armed)
    self.assertTrue(state.lateral_off)
    press_lkas(state, parsers, 1_020_000_000, 1)
    self.assertFalse(state.lka_armed)
    self.assertTrue(state.lateral_off)

    # Dynamic transition across settings changes.
    state, parsers = init_state(True, True, last_centering=False)
    self.assertTrue(state.lka_armed)
    state.lka_enabled = False
    state.update(parsers)
    self.assertFalse(state.lka_armed)
    self.assertFalse(state.lateral_off)
    state.aol_enabled = False
    state.lka_enabled = True
    state.update(parsers)
    self.assertTrue(state.lka_armed)
    self.assertFalse(state.lateral_off)

  def test_carcontroller_dash_state_and_steering_across_lateral_modes(self):
    _, controller, command, state = setup_controller(CAR.SUBARU_LEGACY_2025)
    state.out.cruiseState.available = True
    state.out.gearShifter = structs.CarState.GearShifter.drive
    state.out.vEgo = 20.0
    state.out.standstill = False

    # Centering active -> LKAS_Dash_State == 2 and ES_LKAS_ANGLE LKAS_Request == 1.
    state.lka_armed = False
    state.lateral_off = False
    command.enabled = True
    command.latActive = True
    command.actuators.steeringAngleDeg = 10.0
    _, angle = step(controller, command, state)
    self.assertEqual(angle['LKAS_Request'], 1)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 2)

    # LKA armed (no active nudge) -> LKAS_Dash_State == 1.
    state.lka_armed = True
    state.lateral_off = False
    command.enabled = False
    command.latActive = False
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)

    # lateral_off with at least one toggle ON -> LKAS_Dash_State == 1 (white ready wheel) and LKAS_Request == 0.
    state.lka_armed = False
    state.lateral_off = True
    state.aol_enabled = True
    state.lka_enabled = False
    command.enabled = True
    command.latActive = True
    _, angle = step(controller, command, state)
    self.assertEqual(angle['LKAS_Request'], 0)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)

    # Both AOL and LKA toggles OFF -> LKAS_Dash_State == 0 (wheel icon off).
    state.aol_enabled = False
    state.lka_enabled = False
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 0)

    # Main cruise OFF -> LKAS_Dash_State == 0 (wheel icon off) even with AOL toggle ON.
    state.aol_enabled = True
    state.out.cruiseState.available = False
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 0)

  def test_lkas_button_hold_survives_conflation_and_keeps_dash_wheel_in_sync(self):
    cp, controller, command, state = setup_controller(CAR.SUBARU_LEGACY_2025)
    _, frames = pair.TestSubaruGen2AnglePair.sources(CAR.SUBARU_LEGACY_2025)
    packer = CANPacker(DBC[CAR.SUBARU_LEGACY_2025][Bus.pt])
    state.aol_enabled = True
    state.lka_enabled = True
    state.last_lateral_mode_centering = True
    parsers = state.get_can_parsers(cp)
    state.update(parsers)
    for parser in parsers.values():
      parser.update([(1_000_000_000, frames)])
    state.update(parsers)
    self.assertFalse(state.lka_armed)

    controlsd_lka_armed = False
    prev_lkas_pressed = False

    def step_controlsd_from_ret(car_state):
      nonlocal controlsd_lka_armed, prev_lkas_pressed
      raw_pressed = any(
        be.type == structs.CarState.ButtonEvent.Type.lkas and be.pressed
        for be in car_state.buttonEvents
      )
      if not car_state.cruiseState.available:
        prev_lkas_pressed = False
      lkas_pressed = raw_pressed and not prev_lkas_pressed
      prev_lkas_pressed = raw_pressed
      if lkas_pressed:
        controlsd_lka_armed = not controlsd_lka_armed

    # 1. Press LKAS (0 -> 1): simulate SubMaster dropping the first two carState frames.
    parsers[Bus.cam].update([(1_020_000_000, [packer.make_can_msg('ES_LKAS_State', CanBus.camera, {'LKAS_Dash_State': 1})])])
    cs_frames = [state.update(parsers) for _ in range(6)]
    self.assertEqual([any(be.type == structs.CarState.ButtonEvent.Type.lkas for be in f.buttonEvents) for f in cs_frames],
                     [True, True, True, True, True, False])
    self.assertTrue(state.lka_armed)
    # Conflated consumer wakes on frame index 2 (dropping frames 0 and 1), then sees frames 3..5.
    for f in cs_frames[2:]:
      step_controlsd_from_ret(f)
    self.assertTrue(controlsd_lka_armed)
    self.assertEqual(state.lka_armed, controlsd_lka_armed)

    # During an active LKA nudge, wheel icon must remain white (1).
    state.out.cruiseState.available = True
    state.out.gearShifter = structs.CarState.GearShifter.drive
    state.out.vEgo = 20.0
    state.out.standstill = False
    command.enabled = False
    command.latActive = True
    command.hudControl.leftLaneDepart = True
    command.actuators.steeringAngleDeg = -8.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)

    # 2. Press LKAS (1 -> 0) back to AOL: again drop first two carState frames via conflation.
    parsers[Bus.cam].update([(1_100_000_000, [packer.make_can_msg('ES_LKAS_State', CanBus.camera, {'LKAS_Dash_State': 0})])])
    cs_frames_2 = [state.update(parsers) for _ in range(6)]
    self.assertEqual([any(be.type == structs.CarState.ButtonEvent.Type.lkas for be in f.buttonEvents) for f in cs_frames_2],
                     [True, True, True, True, True, False])
    self.assertFalse(state.lka_armed)
    for f in cs_frames_2[2:]:
      step_controlsd_from_ret(f)
    self.assertFalse(controlsd_lka_armed)
    self.assertEqual(state.lka_armed, controlsd_lka_armed)

    # In AOL mode with active steering, wheel icon turns back to green (2).
    command.enabled = True
    command.latActive = True
    command.hudControl.lanesVisible = True
    command.hudControl.leftLaneDepart = False
    command.actuators.steeringAngleDeg = 8.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    self.assertEqual(decoded['SET_3'], 3)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 2)

  def test_carcontroller_aol_green_wheel_in_park_and_steer_required_chime(self):
    _, controller, command, state = setup_controller(CAR.SUBARU_LEGACY_2025)
    state.out.cruiseState.available = True
    state.out.gearShifter = structs.CarState.GearShifter.park
    state.out.vEgo = 0.0
    state.out.standstill = True
    state.aol_enabled = True
    state.lka_enabled = True

    # AOL active in Park / standstill (latActive=False, lanesVisible=True) -> green wheel icon (2).
    state.lka_armed = False
    state.lateral_off = False
    command.enabled = False
    command.latActive = False
    command.hudControl.lanesVisible = True
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 2)

    # Toggle LKAS to LKA mode in Park -> immediately drops to white wheel icon (1).
    state.lka_armed = True
    command.hudControl.lanesVisible = False
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)

    # LKA hands-on-wheel stages on 0x322: 5s (visual only: 1/0), 10s (+bouncy chime: 1/28), 15s lockout (7/30).
    command.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.steerRequired
    command.hudControl.audibleAlert = structs.CarControl.HUDControl.AudibleAlert.none
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 1)
    self.assertEqual(hud['LKAS_Alert'], 0)

    command.hudControl.audibleAlert = structs.CarControl.HUDControl.AudibleAlert.warningSoft
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 7)
    self.assertEqual(hud['LKAS_Alert'], 28)
    # 10s chime pulses off cleanly at 1.4s..1.99s while keeping flashing red hands active, then re-triggers at 2.0s.
    while controller.lka_chime_pulse_ticks < 145:
      hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 7)
    self.assertEqual(hud['LKAS_Alert'], 0)
    while controller.lka_chime_pulse_ticks < 205:
      hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 7)
    self.assertEqual(hud['LKAS_Alert'], 28)

    command.hudControl.audibleAlert = structs.CarControl.HUDControl.AudibleAlert.warningImmediate
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 7)
    self.assertEqual(hud['LKAS_Alert'], 30)
    # 15s lockout chime pulses off cleanly at 2.2s..2.99s while keeping escalated border/message active, then re-triggers at 3.0s.
    while controller.lka_chime_pulse_ticks < 225:
      hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 7)
    self.assertEqual(hud['LKAS_Alert'], 0)
    while controller.lka_chime_pulse_ticks < 305:
      hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 7)
    self.assertEqual(hud['LKAS_Alert'], 30)

    command.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.none
    command.hudControl.audibleAlert = structs.CarControl.HUDControl.AudibleAlert.none
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert_Msg'], 0)
    self.assertEqual(hud['LKAS_Alert'], 0)

    # Driving in AOL / full OP: active steering -> green (2), manual override -> white (1), resume -> green (2).
    state.lka_armed = False
    state.lateral_off = False
    state.out.gearShifter = structs.CarState.GearShifter.drive
    state.out.vEgo = 20.0
    state.out.vEgoRaw = 20.0
    state.out.standstill = False
    command.enabled = True
    command.latActive = True
    command.hudControl.lanesVisible = True
    state.out.steeringTorque = 0.0
    state.out.steeringRateDeg = 0.0
    state.out.steeringPressed = False
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    self.assertEqual(decoded['SET_3'], 3)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 2)

    # Driver manually overrides steering -> wheel icon turns white (1).
    state.out.steeringTorque = 250.0
    state.out.steeringRateDeg = 30.0
    state.out.steeringPressed = True
    step(controller, command, state)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)

    # Driver releases wheel -> openpilot resumes control and wheel icon turns back to green (2).
    state.out.steeringTorque = 0.0
    state.out.steeringRateDeg = 0.0
    state.out.steeringPressed = False
    step(controller, command, state)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    self.assertEqual(decoded['SET_3'], 3)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 2)

    # Stock EyeSight LDW / allowed alert takes priority over LKA steerRequired alert.
    state.es_lkas_state_msg["LKAS_Alert"] = 12
    command.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.steerRequired
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Alert'], 12)
    self.assertEqual(hud['LKAS_Alert_Msg'], 0)

    dbc = DBC[CAR.SUBARU_LEGACY_2025][Bus.pt]
    packer = CANPacker(dbc)
    info_parser = CANParser(dbc, [('ES_Infotainment', 0)], CanBus.main)
    info_msg = subarucan.create_es_infotainment(
      packer, 0, {'CHECKSUM': 0, 'LKAS_State_Infotainment': 1, 'LKAS_Blue_Lines': 0, 'Signal1': 0, 'Signal2': 0},
      structs.CarControl.HUDControl.VisualAlert.steerRequired,
      legacy_2025=True,
    )
    info_parser.update([(1, [info_msg])])
    self.assertEqual(info_parser.vl['ES_Infotainment']['LKAS_State_Infotainment'], 1)

    for audible, expected in (
      (structs.CarControl.HUDControl.AudibleAlert.none, 0),
      (structs.CarControl.HUDControl.AudibleAlert.warningSoft, 3),
      (structs.CarControl.HUDControl.AudibleAlert.warningImmediate, 4),
    ):
      info_msg = subarucan.create_es_infotainment(
        packer, 1, {'CHECKSUM': 0, 'LKAS_State_Infotainment': 0, 'LKAS_Blue_Lines': 0, 'Signal1': 0, 'Signal2': 0},
        structs.CarControl.HUDControl.VisualAlert.steerRequired,
        audible_alert=audible, legacy_2025=True,
      )
      info_parser.update([(2, [info_msg])])
      self.assertEqual(info_parser.vl['ES_Infotainment']['LKAS_State_Infotainment'], expected)

    # Non-Legacy default (legacy_2025=False) preserves exact upstream 0x323 behavior.
    for stock_state in (0, 1):
      info_msg = subarucan.create_es_infotainment(
        packer, 2, {'CHECKSUM': 0, 'LKAS_State_Infotainment': stock_state, 'LKAS_Blue_Lines': 0, 'Signal1': 0, 'Signal2': 0},
        structs.CarControl.HUDControl.VisualAlert.steerRequired,
        audible_alert=structs.CarControl.HUDControl.AudibleAlert.warningImmediate,
      )
      info_parser.update([(3, [info_msg])])
      self.assertEqual(info_parser.vl['ES_Infotainment']['LKAS_State_Infotainment'], 3)

  def test_carcontroller_lka_manumatic_and_unauthorized_gate(self):
    _, controller, command, state = setup_controller(CAR.SUBARU_LEGACY_2025)
    state.out.gearShifter = structs.CarState.GearShifter.manumatic
    state.lka_armed = True
    state.out.steeringAngleDeg = 0.0
    state.out.steeringRateDeg = 0.0
    state.out.steeringTorque = 0.0

    # Standby HUD in manumatic: no torque/angle request, white wheel icon (1), visible left line (1).
    command.enabled = False
    command.latActive = False
    command.hudControl.leftLaneVisible = True
    command.hudControl.rightLaneVisible = False
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 1)

    # Active LKA nudge in manumatic: angle requested, white wheel icon (1), amber left line (3).
    command.latActive = True
    command.hudControl.leftLaneDepart = True
    command.actuators.steeringAngleDeg = -8.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    self.assertLess(decoded['LKAS_Output'], 0.0)
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 3)

    # Unauthorized gate: when controlsd keeps latActive=False, CarController must never request 0x124 torque/angle.
    command.latActive = False
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)

  def test_legacy_2025_manumatic_transition_and_between_detent_gears(self):
    cp = CarInterface.get_params(CAR.SUBARU_LEGACY_2025, {0: {}, 1: {}, 2: {}}, [], False, False, False)
    _, frames = pair.TestSubaruGen2AnglePair.sources(CAR.SUBARU_LEGACY_2025)
    packer = CANPacker(DBC[CAR.SUBARU_LEGACY_2025][Bus.pt])
    state = CarState(cp)
    parsers = state.get_can_parsers(cp)
    state.update(parsers)
    for parser in parsers.values():
      parser.update([(1_000_000_000, frames)])
    state.update(parsers)

    Gear = structs.CarState.GearShifter
    for can_gear, expected_gear in ((121, Gear.drive), (129, Gear.manumatic), (177, Gear.manumatic), (0, Gear.neutral), (4, Gear.park)):
      msg = packer.make_can_msg('Transmission', 0, {'Gear': can_gear})
      parsers[Bus.pt].update([(1_020_000_000, [msg])])
      ret = state.update(parsers)
      self.assertEqual(ret.gearShifter, expected_gear, f'can_gear={can_gear}')

  def test_legacy_2025_acc_lead_led_preserved_and_asymmetric_aol_line_debounce(self):
    _, _, command, state = setup_controller(CAR.SUBARU_LEGACY_2025)
    cp = CarInterface.get_params(CAR.SUBARU_LEGACY_2025, {0: {}, 1: {}, 2: {0x3C4: 8}}, [], False, False, False)
    controller = CarController(DBC[CAR.SUBARU_LEGACY_2025], cp)
    state.out.cruiseState.available = True
    state.out.gearShifter = structs.CarState.GearShifter.drive
    state.out.vEgo = 20.0
    state.out.standstill = False

    # Stock ACC active via Cruise_Activated_Dash + Car_Follow (not gasPressed) preserves dat[5]=0x01 when masking 000104000000.
    state.out.cruiseState.enabled = True
    state.out.gasPressed = False
    state.es_dashstatus_msg.update({'Cruise_Activated_Dash': 1, 'Cruise_State': 0, 'Car_Follow': 1})
    raw = bytearray(bytes.fromhex('0002000104000000'))
    raw[0] = (0xC4 + 3 + sum(raw[1:])) & 0xFF
    state.es_lkas_alert_dat = bytes(raw)
    msg = controller._ascent_lkas_alert(command.as_reader(), state)
    self.assertIsNotNone(msg)
    self.assertEqual(msg[1][2:], bytes.fromhex('000000010000'))

    # Gas pedal override clears green lead LED (dat[5] = 0x00) even when cruiseState.enabled and Car_Follow remain set.
    for idx, override_fields in enumerate((
      {'gas': True, 'dash': 1, 'cstate': 0},
      {'gas': False, 'dash': 0, 'cstate': 2},
    )):
      state.out.gasPressed = override_fields['gas']
      state.es_dashstatus_msg.update({
        'Cruise_Activated_Dash': override_fields['dash'],
        'Cruise_State': override_fields['cstate'],
        'Car_Follow': 1,
      })
      raw[1] = 0x03 + idx
      raw[0] = (0xC4 + 3 + sum(raw[1:])) & 0xFF
      state.es_lkas_alert_dat = bytes(raw)
      msg = controller._ascent_lkas_alert(command.as_reader(), state)
      self.assertIsNotNone(msg)
      self.assertEqual(msg[1][2:], bytes.fromhex('000000000000'), override_fields)
    state.out.gasPressed = False
    state.es_dashstatus_msg.update({'Cruise_Activated_Dash': 1, 'Cruise_State': 0})

    # Without Car_Follow, dat[5] stays 0x00 when masking the alert.
    state.es_dashstatus_msg.update({'Cruise_Activated_Dash': 1, 'Cruise_State': 0, 'Car_Follow': 0})
    raw[1] = 0x05
    raw[0] = (0xC4 + 3 + sum(raw[1:])) & 0xFF
    state.es_lkas_alert_dat = bytes(raw)
    msg = controller._ascent_lkas_alert(command.as_reader(), state)
    self.assertIsNotNone(msg)
    self.assertEqual(msg[1][2:], bytes.fromhex('000000000000'))

    # AOL mode passes debounced hudControl lane visibility directly through to cluster CAN.
    state.lka_armed = False
    state.lateral_off = False
    command.enabled = False
    command.latActive = True
    command.hudControl.leftLaneVisible = True
    command.hudControl.rightLaneVisible = True
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 1)
    self.assertEqual(hud['LKAS_Right_Line_Visible'], 1)

    command.hudControl.leftLaneVisible = False
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 0)
    self.assertEqual(hud['LKAS_Right_Line_Visible'], 1)

    command.hudControl.leftLaneVisible = True
    hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Left_Line_Visible'], 1)

    # In LKA nudge mode, same-direction (assisting) torque > 175 overrides equally to opposing torque.
    state.lka_armed = True
    command.latActive = True
    command.actuators.steeringAngleDeg = -8.0
    state.out.steeringAngleDeg = 0.0
    state.out.steeringRateDeg = 0.0
    state.out.steeringTorque = 0.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    state.out.steeringTorque = -150.0
    step(controller, command, state)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    state.out.steeringTorque = -180.0
    step(controller, command, state)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)

    # Pre-latched curve-holding assisting torque (-175) allows outside-line LKA nudge;
    # pushing >235 total or opposing >175 still overrides cleanly.
    command.latActive = False
    command.hudControl.leftLaneDepart = False
    state.out.steeringAngleDeg = -10.0
    state.out.steeringRateDeg = 0.0
    state.out.steeringTorque = -175.0
    for _ in range(60):
      step(controller, command, state)
    command.latActive = True
    command.hudControl.leftLaneDepart = True
    command.actuators.steeringAngleDeg = -13.0
    state.out.steeringTorque = -183.0
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 1)
    state.out.steeringTorque = -240.0
    step(controller, command, state)
    _, decoded = step(controller, command, state)
    self.assertEqual(decoded['LKAS_Request'], 0)

  def test_aol_wheel_icon_park_reverse_neutral_and_shift_debounce(self):
    _, controller, command, state = setup_controller(CAR.SUBARU_LEGACY_2025)
    state.out.cruiseState.available = True
    state.out.vEgo = 0.0
    state.out.vEgoRaw = 0.0
    state.out.standstill = True
    state.aol_enabled = True
    state.lka_enabled = True
    state.lka_armed = False
    state.lateral_off = False
    command.enabled = False
    command.latActive = False
    command.hudControl.lanesVisible = True

    # In Park: stays solid green even if steering wheel is moved slightly.
    state.out.gearShifter = structs.CarState.GearShifter.park
    state.out.steeringPressed = True
    self.assertEqual(_step_to_hud(controller, command, state)['LKAS_Dash_State'], 2)

    # Fast Park -> Reverse -> Neutral -> Drive shift (< 0.8s) holds green without flashing white.
    state.out.steeringPressed = False
    for gear, steps in (
      (structs.CarState.GearShifter.reverse, 3),
      (structs.CarState.GearShifter.neutral, 3),
    ):
      state.out.gearShifter = gear
      for _ in range(steps):
        self.assertEqual(_step_to_hud(controller, command, state)['LKAS_Dash_State'], 2)
    state.out.gearShifter = structs.CarState.GearShifter.drive
    self.assertEqual(_step_to_hud(controller, command, state)['LKAS_Dash_State'], 2)

    # Settling in Reverse (>= 0.8s) switches to solid white regardless of steering input.
    state.out.gearShifter = structs.CarState.GearShifter.reverse
    for _ in range(9):
      hud = _step_to_hud(controller, command, state)
    self.assertEqual(hud['LKAS_Dash_State'], 1)
    state.out.steeringPressed = True
    self.assertEqual(_step_to_hud(controller, command, state)['LKAS_Dash_State'], 1)
    state.out.gearShifter = structs.CarState.GearShifter.neutral
    self.assertEqual(_step_to_hud(controller, command, state)['LKAS_Dash_State'], 1)

    # Returning to Park restores solid green cleanly.
    state.out.gearShifter = structs.CarState.GearShifter.park
    self.assertEqual(_step_to_hud(controller, command, state)['LKAS_Dash_State'], 2)


if __name__ == '__main__':
  unittest.main()
