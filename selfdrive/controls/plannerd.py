#!/usr/bin/env python3
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.lanekeep import (
  LaneKeepController, LaneKeepMode, LaneKeepSensitivity,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
import cereal.messaging as messaging

from openpilot.starpilot.common.starpilot_variables import get_starpilot_toggles


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  lka = LaneKeepController()
  longitudinal_planner = LongitudinalPlanner(CP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'radarState', 'modelV2', 'selfdriveState'],
                           poll='modelV2')

  sm = sm.extend(['starpilotCarState', 'starpilotPlan'])

  starpilot_toggles = get_starpilot_toggles()

  while True:
    sm.update()
    if sm.updated['modelV2']:
      longitudinal_planner.update(sm, starpilot_toggles)
      longitudinal_planner.publish(sm, pm)

      lka_enabled = getattr(starpilot_toggles, "lane_keep_assist", True)
      lka_mode = getattr(starpilot_toggles, "lane_keep_mode", LaneKeepMode.LANE_LINE_AND_ROAD_EDGE)
      lka_sensitivity = getattr(starpilot_toggles, "lane_keep_sensitivity", LaneKeepSensitivity.NORMAL)
      lka_min_speed = getattr(starpilot_toggles, "lane_keep_min_speed", 25.0)

      cs = sm['carState']
      cc = sm['carControl']

      lka.update(
        model_v2=sm['modelV2'],
        v_ego=cs.vEgo,
        steering_angle_deg=cs.steeringAngleDeg,
        steering_torque=getattr(cs, "steeringTorque", 0.0),
        left_blinker=cs.leftBlinker,
        right_blinker=cs.rightBlinker,
        lat_active=cc.latActive,
        mode=lka_mode if lka_enabled else LaneKeepMode.OFF,
        sensitivity=lka_sensitivity,
        min_activation_speed_mph=lka_min_speed,
        dt=0.05,
      )

      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(['carState', 'carControl', 'modelV2', 'liveParameters'])
      msg.driverAssistance.leftLaneDeparture = bool(lka.left_line_state == 3)
      msg.driverAssistance.rightLaneDeparture = bool(lka.right_line_state == 3)
      pm.send('driverAssistance', msg)

    starpilot_toggles = get_starpilot_toggles(sm)


if __name__ == "__main__":
  main()
