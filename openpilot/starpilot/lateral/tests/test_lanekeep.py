import math
from types import SimpleNamespace
import unittest

from openpilot.selfdrive.controls.lib.lanekeep import DT_CTRL, LaneKeepAlert, LaneKeepController, _update_line_visibility


def _line(y0: float):
  xs = [0.0, 10.0, 20.0, 30.0]
  ys = [y0] * 4
  return SimpleNamespace(x=xs, y=ys)


def _model(left_y: float = -1.8, right_y: float = 1.8, left_prob: float = 0.9, right_prob: float = 0.9,
           desired_curvature: float = 0.0):
  return SimpleNamespace(
    laneLineProbs=[0.0, left_prob, right_prob, 0.0],
    laneLines=[_line(-3.6), _line(left_y), _line(right_y), _line(3.6)],
    roadEdges=[_line(-3.0), _line(3.0)],
    roadEdgeStds=[0.2, 0.2],
    action=SimpleNamespace(desiredCurvature=desired_curvature),
  )


class TestLaneKeepController(unittest.TestCase):
  def test_standby_and_left_boundary_trigger(self):
    lka = LaneKeepController()
    # Centered in lane at 30 mph (~13.4 m/s): standby with white lines (1).
    lka.update(_model(left_y=-1.8, right_y=1.8), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertFalse(lka.active)
    self.assertEqual(lka.left_line_state, 1)
    self.assertEqual(lka.right_line_state, 1)
    self.assertEqual(lka.desired_curvature, 0.0)

    # Drift right next to left painted line (d_left = -left_y - 0.96 < 0.07): triggers amber left nudge (>0 curvature right).
    lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertTrue(lka.active)
    self.assertEqual(lka.left_line_state, 3)
    self.assertGreater(lka.desired_curvature, 0.0)
    self.assertTrue(math.isfinite(lka.desired_curvature))

    # MADS/ACC active centering takes priority and resets LKA.
    lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0, lat_active=True)
    self.assertFalse(lka.active)
    self.assertEqual(lka.left_line_state, 0)
    self.assertEqual(lka.desired_curvature, 0.0)

  def test_line_visibility_threshold_and_asymmetric_debounce(self):
    lka = LaneKeepController()
    # Below 0.55 threshold stays invisible on startup.
    lka.update(_model(left_prob=0.50, right_prob=0.50), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.left_line_state, 0)
    self.assertEqual(lka.right_line_state, 0)

    # Fresh controller at >= 0.55 becomes visible immediately; hysteresis [0.40, 0.55) keeps it visible.
    lka = LaneKeepController()
    lka.update(_model(left_prob=0.55, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.left_line_state, 1)
    self.assertEqual(lka.right_line_state, 1)

    for _ in range(30):
      lka.update(_model(left_prob=0.45, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
      self.assertEqual(lka.left_line_state, 1)

    # Dropping below 0.40 holds for 0.25s (25 ticks) before dropping, and brief recovery resets the off timer.
    for _ in range(20):
      lka.update(_model(left_prob=0.39, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
      self.assertEqual(lka.left_line_state, 1)
    lka.update(_model(left_prob=0.45, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.left_line_state, 1)

    for _ in range(24):
      lka.update(_model(left_prob=0.35, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
      self.assertEqual(lka.left_line_state, 1)
    lka.update(_model(left_prob=0.35, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.left_line_state, 0)
    self.assertEqual(lka.right_line_state, 1)

    # Once hidden, probs in [0.40, 0.55) stay hidden and >= 0.55 must hold for 1.0s (100 ticks) to reappear.
    for _ in range(10):
      lka.update(_model(left_prob=0.50, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
      self.assertEqual(lka.left_line_state, 0)
    for _ in range(99):
      lka.update(_model(left_prob=0.65, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
      self.assertEqual(lka.left_line_state, 0)
    lka.update(_model(left_prob=0.65, right_prob=0.60), v_ego=13.4, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.left_line_state, 1)

  def test_line_offset_shifts_trigger_boundary(self):
    # At left_y=-1.04 (d_left = 0.08m >= 0.07m): default offset 0 stays standby; +2 in (+0.0508m) triggers early.
    lka = LaneKeepController()
    lka.update(_model(left_y=-1.04, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0, line_offset_in=0.0)
    self.assertFalse(lka.active)
    self.assertEqual(lka.left_line_state, 1)

    lka.update(_model(left_y=-1.04, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0, line_offset_in=2.0)
    self.assertTrue(lka.active)
    self.assertEqual(lka.left_line_state, 3)

    # At left_y=-0.99 (d_left = 0.03m < 0.07m): default offset triggers; -3 in (-0.0762m) delays assist.
    lka_neg = LaneKeepController()
    lka_neg.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0, line_offset_in=-3.0)
    self.assertFalse(lka_neg.active)
    self.assertEqual(lka_neg.left_line_state, 1)

    lka_neg.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0, line_offset_in=0.0)
    self.assertTrue(lka_neg.active)
    self.assertEqual(lka_neg.left_line_state, 3)

  def test_inward_driver_torque_releases_immediately(self):
    lka = LaneKeepController()
    for _ in range(5):
      lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertTrue(lka.active)
    self.assertEqual(lka.left_line_state, 3)
    self.assertGreater(lka.desired_curvature, 0.0)

    # Driver steers right (negative CAN torque, inward away from left line) above 65 CAN units for 50 ms.
    for _ in range(5):
      lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=-70.0)
    self.assertFalse(lka.active)
    self.assertEqual(lka.desired_curvature, 0.0)
    # Amber line visual hold remains active without holding actuator torque.
    self.assertEqual(lka.left_line_state, 3)

    # Releasing inward torque resumes immediately on the next step with zero cooldown.
    lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertTrue(lka.active)
    self.assertGreater(lka.desired_curvature, 0.0)

    # Opposing outward torque <= 175 does not override; > 175 triggers cooldown override.
    for _ in range(6):
      lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=150.0)
    self.assertTrue(lka.active)
    for _ in range(6):
      lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=180.0)
    self.assertFalse(lka.active)
    lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertFalse(lka.active)  # Latched 1.20s cooldown holds after outward override

  def test_curve_holding_allows_outside_line_nudge_and_releases_on_extra_inward_torque(self):
    lka = LaneKeepController()
    # Driver holds steady -175 CAN torque around a right curve before drifting onto the outside left line.
    for _ in range(40):
      lka.update(_model(left_y=-1.4, right_y=2.2, desired_curvature=0.0045),
                 v_ego=18.5, steering_angle_deg=-10.0, steering_torque=-175.0)
    self.assertFalse(lka.active)

    # Crossing left line while still holding ~-180 CAN torque engages outside-line assist instead of locking out.
    for _ in range(5):
      lka.update(_model(left_y=-0.99, right_y=2.6, desired_curvature=0.0045),
                 v_ego=18.5, steering_angle_deg=-10.5, steering_torque=-180.0)
    self.assertTrue(lka.active)
    self.assertGreater(lka.desired_curvature, 0.0)
    self.assertLess(lka.assist_torque_baseline, -150.0)

    # Adding >55 CAN torque further inward on top of the held curve baseline releases immediately.
    for _ in range(5):
      lka.update(_model(left_y=-0.99, right_y=2.6, desired_curvature=0.0045),
                 v_ego=18.5, steering_angle_deg=-11.0, steering_torque=-240.0)
    self.assertFalse(lka.active)
    self.assertEqual(lka.desired_curvature, 0.0)

    # Mirror case: left curve holding +150 CAN torque while drifting wide across the right line.
    lka_right = LaneKeepController()
    for _ in range(40):
      lka_right.update(_model(left_y=-2.2, right_y=1.4, desired_curvature=-0.0035),
                       v_ego=16.0, steering_angle_deg=10.0, steering_torque=150.0)
    for _ in range(5):
      lka_right.update(_model(left_y=-2.6, right_y=0.91, desired_curvature=-0.0035),
                       v_ego=16.0, steering_angle_deg=10.5, steering_torque=165.0)
    self.assertTrue(lka_right.active)
    self.assertLess(lka_right.desired_curvature, 0.0)

  def test_mild_inside_apex_buffer(self):
    # Gentle bend (curv 0.0012 < 0.002) keeps normal +0.07m margin and catches at d_right=+0.05m (right_y=0.93).
    lka_gentle = LaneKeepController()
    for _ in range(3):
      lka_gentle.update(_model(left_y=-2.6, right_y=0.93, desired_curvature=0.0012),
                        v_ego=18.0, steering_angle_deg=-3.0, steering_torque=0.0)
    self.assertTrue(lka_gentle.active)

    # Real curve (curv 0.003 > 0.002) gives 0.06m apex allowance (+0.07 -> +0.01):
    # stays quiet at d_right=+0.03m (right_y=0.91), triggers cleanly once at d_right=0.00m (right_y=0.88).
    lka_curve = LaneKeepController()
    for _ in range(3):
      lka_curve.update(_model(left_y=-2.6, right_y=0.91, desired_curvature=0.003),
                       v_ego=18.0, steering_angle_deg=-8.0, steering_torque=0.0)
    self.assertFalse(lka_curve.active)
    for _ in range(3):
      lka_curve.update(_model(left_y=-2.6, right_y=0.88, desired_curvature=0.003),
                       v_ego=18.0, steering_angle_deg=-8.0, steering_torque=0.0)
    self.assertTrue(lka_curve.active)

  def test_aol_shared_line_visibility_debounce_timing(self):
    vis, timer = _update_line_visibility(None, 0.0, 0.60, DT_CTRL)
    self.assertTrue(vis)

    # Brief <0.40 dip under 0.25s holds True and resets timer on recovery into [0.40, 0.55).
    for _ in range(24):
      vis, timer = _update_line_visibility(vis, timer, 0.35, DT_CTRL)
      self.assertTrue(vis)
    vis, timer = _update_line_visibility(vis, timer, 0.45, DT_CTRL)
    self.assertTrue(vis)
    self.assertEqual(timer, 0.0)

    # Sustained <0.40 for 0.25s drops cleanly, then requires 1.0s at >=0.55 to return.
    for _ in range(24):
      vis, timer = _update_line_visibility(vis, timer, 0.35, DT_CTRL)
      self.assertTrue(vis)
    vis, timer = _update_line_visibility(vis, timer, 0.35, DT_CTRL)
    self.assertFalse(vis)

    for _ in range(99):
      vis, timer = _update_line_visibility(vis, timer, 0.60, DT_CTRL)
      self.assertFalse(vis)
    vis, timer = _update_line_visibility(vis, timer, 0.60, DT_CTRL)
    self.assertTrue(vis)

  def test_hands_off_escalation_5s_10s_15s_and_reset(self):
    lka = LaneKeepController()
    drifting = _model(left_y=-0.99, right_y=2.6)

    # 0..4.99s: active nudging without warning
    for _ in range(499):
      lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.NONE)

    # 5.0s..9.99s: visual-only 5s warning
    lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.KEEP_HANDS_ON_WHEEL)
    for _ in range(499):
      lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.KEEP_HANDS_ON_WHEEL)

    # 10.0s..14.99s: intermediary soft bouncy chime alert while still active
    lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.KEEP_HANDS_ON_WHEEL_CHIME)
    self.assertTrue(lka.active)
    for _ in range(499):
      lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.KEEP_HANDS_ON_WHEEL_CHIME)

    # 15.0s: lockout stops nudging and escalates to urgent OFF alert
    lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.EMERGENCY_DEPARTURE)
    self.assertFalse(lka.nudge_active)

    # Once smoothed curvature settles at lane center, >= 85 CAN torque for 0.12s clears lockout & alert
    centered = _model(left_y=-1.8, right_y=1.8)
    for _ in range(50):
      lka.update(centered, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertFalse(lka.active)
    for _ in range(15):
      lka.update(centered, v_ego=18.0, steering_angle_deg=0.0, steering_torque=90.0)
    self.assertEqual(lka.alert, LaneKeepAlert.NONE)

  def test_ttc_hilly_noise_at_5_to_6_inches_suppressed_while_line_margin_triggers(self):
    lka = LaneKeepController()
    # Brief vision horizon shift (~0.10s) moving from d_r=+0.21m -> +0.15m stays quiet (~6 inches off line).
    for _ in range(5):
      lka.update(_model(left_y=-1.40, right_y=1.09), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0, model_updated=True)
    for tick in range(10):
      lka.update(_model(left_y=-1.40, right_y=1.03), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0, model_updated=(tick % 5 == 0))
    self.assertFalse(lka.active)
    self.assertEqual(lka.right_line_state, 1)

    # Real drift inside standard margin (d_r = +0.05m < 0.07m) still triggers immediately.
    lka.update(_model(left_y=-1.40, right_y=0.93), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0, model_updated=True)
    self.assertTrue(lka.active)
    self.assertEqual(lka.right_line_state, 3)

  def test_encroached_vision_jitter_stays_smooth_and_reaches_15s_lockout(self):
    lka = LaneKeepController()
    # Tire stays outside the left line (-0.88 .. -0.92m) with 20 Hz +-2cm vision jitter causing d_dot swings > +0.05 m/s.
    lockout_step = None
    for tick in range(1550):
      phase = (tick // 5) % 2
      left_y = -0.88 if phase == 0 else -0.92
      lka.update(_model(left_y=left_y, right_y=2.6), v_ego=20.0, steering_angle_deg=0.0,
                 steering_torque=0.0, model_updated=(tick % 5 == 0))
      if lka.alert == LaneKeepAlert.EMERGENCY_DEPARTURE:
        lockout_step = tick
        break
      self.assertTrue(lka.active, f"pulsed off at tick {tick}")
      self.assertGreater(lka.desired_curvature, 0.0)
    self.assertIsNotNone(lockout_step)
    self.assertLessEqual(lockout_step, 1505)

  def test_mid_departure_light_drag_does_not_release_straighten(self):
    lka = LaneKeepController()
    # Drift deep across left line with light hand drag (+77 CAN) and small +0.03 m/s noise (seg 22 scenario).
    for tick in range(25):
      left_y = -0.635 + (0.0015 if tick % 5 == 0 else 0.0)
      lka.update(_model(left_y=left_y, right_y=2.8), v_ego=17.5, steering_angle_deg=-3.0,
                 steering_torque=77.0, model_updated=(tick % 5 == 0))
      self.assertTrue(lka.active)
      self.assertGreater(lka.desired_curvature, 0.0)

    # Once deliberately holding > 95 CAN torque after recovering back near d_thresh, straighten releases cleanly.
    for tick in range(10):
      lka.update(_model(left_y=-1.01 + tick * 0.002, right_y=2.6), v_ego=17.5, steering_angle_deg=0.0,
                 steering_torque=100.0, model_updated=(tick % 5 == 0))
    self.assertFalse(lka.active)


if __name__ == "__main__":
  unittest.main()
