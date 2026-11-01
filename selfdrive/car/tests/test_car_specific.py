from types import SimpleNamespace

from cereal import car
from opendbc.car.hyundai.values import CAR as HYUNDAI_CAR

from openpilot.selfdrive.car.car_specific import CarSpecificEvents
from openpilot.selfdrive.car.cruise_state import should_cancel_stock_cruise, should_flag_cruise_mismatch


def make_cp(brand="hyundai", op_long=True, pcm_cruise=False):
  cp = car.CarParams.new_message()
  cp.brand = brand
  cp.openpilotLongitudinalControl = op_long
  cp.pcmCruise = pcm_cruise
  return cp


def test_hyundai_openpilot_long_does_not_cancel_active_acc_req_feedback():
  cp = make_cp()

  assert not should_cancel_stock_cruise(cp, cruise_enabled=True, controls_enabled=True)
  assert not should_flag_cruise_mismatch(cp, cruise_enabled=True, controls_enabled=True, effective_pcm_cruise=False)


def test_hyundai_openpilot_long_still_flags_cruise_when_controls_disabled():
  cp = make_cp()

  assert should_cancel_stock_cruise(cp, cruise_enabled=True, controls_enabled=False)
  assert should_flag_cruise_mismatch(cp, cruise_enabled=True, controls_enabled=False, effective_pcm_cruise=False)


def test_non_hyundai_openpilot_long_behavior_is_unchanged():
  cp = make_cp(brand="toyota")

  assert should_cancel_stock_cruise(cp, cruise_enabled=True, controls_enabled=True)
  assert should_flag_cruise_mismatch(cp, cruise_enabled=True, controls_enabled=True, effective_pcm_cruise=False)


def test_pcm_cruise_behavior_is_unchanged():
  cp = make_cp(op_long=False, pcm_cruise=True)

  assert not should_cancel_stock_cruise(cp, cruise_enabled=True, controls_enabled=True)
  assert should_cancel_stock_cruise(cp, cruise_enabled=True, controls_enabled=False)
  assert not should_flag_cruise_mismatch(cp, cruise_enabled=True, controls_enabled=True, effective_pcm_cruise=True)
  assert should_flag_cruise_mismatch(cp, cruise_enabled=True, controls_enabled=False, effective_pcm_cruise=True)


def test_kia_ray_ev_allows_stock_cruise_to_enable_controls():
  for candidate, ignore_cruise_state in ((HYUNDAI_CAR.KIA_RAY_EV, True), (HYUNDAI_CAR.HYUNDAI_SONATA, False)):
    cp = SimpleNamespace(brand="hyundai", carFingerprint=candidate, flags=0)
    handler = CarSpecificEvents(cp)
    captured = {}
    handler.create_common_events = lambda *args, **kwargs: captured.update(kwargs)

    handler.update(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())

    assert captured["pcm_enable"] is True
    assert captured["ignore_cruise_state"] is ignore_cruise_state


def test_subaru_legacy_2025_allows_manumatic_paddle_shift_gear_without_wrong_gear_event():
  from cereal import log
  from opendbc.car.subaru.values import CAR as SUBARU_CAR

  GearShifter = car.CarState.GearShifter
  EventName = log.OnroadEvent.EventName

  def run_gear_check(fingerprint, gear):
    cp = car.CarParams.new_message()
    cp.brand = "subaru"
    cp.carFingerprint = fingerprint
    cp.pcmCruise = True
    handler = CarSpecificEvents(cp)

    cs = car.CarState.new_message()
    cs.gearShifter = gear
    cs.cruiseState.available = True
    cs.cruiseState.enabled = True
    cs.doorOpen = False
    cs.seatbeltUnlatched = False

    cs_prev = car.CarState.new_message()
    cs_prev.gearShifter = GearShifter.drive
    cs_prev.cruiseState.available = True
    cs_prev.cruiseState.enabled = True

    cc = car.CarControl.new_message()
    events = handler.update(cs, cs_prev, cc)
    return events

  # 2025 Subaru Legacy allows CVT paddle-shifter manual mode (GearShifter.manumatic) without wrongGear fault
  legacy_manumatic_events = run_gear_check(SUBARU_CAR.SUBARU_LEGACY_2025, GearShifter.manumatic)
  assert EventName.wrongGear not in legacy_manumatic_events.names

  # Park/Reverse/Neutral still emit wrongGear on 2025 Subaru Legacy when engaged
  legacy_reverse_events = run_gear_check(SUBARU_CAR.SUBARU_LEGACY_2025, GearShifter.reverse)
  assert EventName.wrongGear in legacy_reverse_events.names

  # Other Subaru models remain strictly gated and still flag GearShifter.manumatic as wrongGear
  outback_manumatic_events = run_gear_check(SUBARU_CAR.SUBARU_OUTBACK_2023, GearShifter.manumatic)
  assert EventName.wrongGear in outback_manumatic_events.names
