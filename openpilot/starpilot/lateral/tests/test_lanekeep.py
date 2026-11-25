import math
from types import SimpleNamespace
import unittest

from openpilot.selfdrive.controls.lib.lanekeep import (
  DT_CTRL,
  LKA_MAX_INWARD_ACCEL_MPS2,
  LaneKeepAlert,
  LaneKeepController,
  _update_line_visibility,
  compute_lka_guided_curvature,
)


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

    # Releasing inward torque holds brief 0.35s anti-chatter cooldown (34 ticks), then resumes on tick 35.
    for _ in range(34):
      lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
      self.assertFalse(lka.active)
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

  def test_cornering_gate_20deg_and_unwind_cooldown(self):
    lka = LaneKeepController()
    # 19 deg stays below the 20 deg gate: standby lines remain visible and triggers stay armed.
    lka.update(_model(left_y=-1.8, right_y=1.8), v_ego=18.0, steering_angle_deg=19.0, steering_torque=0.0)
    self.assertEqual(lka.left_line_state, 1)
    self.assertEqual(lka.right_line_state, 1)

    # Crossing >= 20 deg hides standby lines and blocks triggers.
    lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=20.0, steering_torque=0.0)
    self.assertFalse(lka.active)
    self.assertEqual(lka.left_line_state, 0)
    self.assertEqual(lka.right_line_state, 0)

    # Unwinding back to 0 deg holds cornering suppression for 1.5s before re-arming.
    for _ in range(149):
      lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
      self.assertFalse(lka.active)
      self.assertEqual(lka.left_line_state, 0)
    lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertTrue(lka.active)
    self.assertEqual(lka.left_line_state, 3)

  def test_receding_entry_guard_blocks_post_turn_unwind_trigger(self):
    lka = LaneKeepController()
    # Pinched near the right line during a suppressed turn, then unwinding cooldown expires while already moving back
    # toward lane center (d_right < 0.07m but d_dot_right >= +0.06 m/s): must not start a fresh post-turn episode.
    for tick in range(150):
      ang = -25.0 if tick == 0 else 0.0
      lka.update(_model(left_y=-2.2, right_y=0.72), v_ego=17.5, steering_angle_deg=ang,
                 steering_torque=0.0, model_updated=(tick % 5 == 0))
    self.assertFalse(lka.active)

    # Cornering cooldown clears on the next tick while d_right is still < 0.07m (right_y=0.92 -> d_right=0.04m)
    # and the car is receding rapidly back into the lane (+0.04m per 50ms vision step = +0.80 m/s raw).
    for tick in range(1, 15):
      y_r = 0.72 + tick * 0.014
      for sub in range(5):
        lka.update(_model(left_y=-2.2, right_y=y_r), v_ego=17.5, steering_angle_deg=0.0,
                   steering_torque=0.0, model_updated=(sub == 0))
        self.assertFalse(lka.active)
        self.assertEqual(lka.right_line_state, 1)

    # Once lateral drift settles (d_dot_right ~ 0) while still within d_thresh (right_y=0.92 -> d_right=0.04m), triggers normally.
    for tick in range(40):
      lka.update(_model(left_y=-2.2, right_y=0.92), v_ego=17.5, steering_angle_deg=0.0,
                 steering_torque=0.0, model_updated=(tick % 5 == 0))
    self.assertTrue(lka.active)
    self.assertEqual(lka.right_line_state, 3)

  def test_guided_curvature_no_ratchet_and_smooth_unwind(self):
    v_ego = 25.0  # ~56 mph
    max_inward_push = LKA_MAX_INWARD_ACCEL_MPS2 / (v_ego * v_ego)

    # 1. Zero ratcheting: anchor stays frozen at entry and guided never exceeds anchor + max_inward_push.
    entry_anchor = -0.0002
    cur_c = entry_anchor
    anchor = entry_anchor
    lka_c = -0.0005
    for _ in range(100):
      guided, anchor = compute_lka_guided_curvature(
        current_curvature=cur_c,
        anchor_curvature=anchor,
        lka_desired_curvature=lka_c,
        d_dot=0.0,
        v_ego=v_ego,
        nudge_right=False,
      )
      self.assertEqual(anchor, entry_anchor)
      self.assertGreaterEqual(guided, entry_anchor - max_inward_push - 1e-9)
      cur_c = guided
    self.assertAlmostEqual(cur_c, entry_anchor + lka_c, places=7)

    # 2. Strictly one-way (0% outward pull) when driver is already steering more inward than anchor + push.
    guided_in, anchor_in = compute_lka_guided_curvature(
      current_curvature=-0.0012,
      anchor_curvature=0.0,
      lka_desired_curvature=-0.0004,
      d_dot=0.0,
      v_ego=v_ego,
      nudge_right=False,
    )
    self.assertEqual(guided_in, -0.0012)
    self.assertEqual(anchor_in, 0.0)

    # 3. Downward-only anchor tracking when driver steers outward keeps push bounded from new outward angle.
    guided_out, anchor_out = compute_lka_guided_curvature(
      current_curvature=0.0003,
      anchor_curvature=0.0,
      lka_desired_curvature=-0.0008,
      d_dot=0.0,
      v_ego=v_ego,
      nudge_right=False,
    )
    self.assertAlmostEqual(anchor_out, 0.0003, places=9)
    self.assertAlmostEqual(guided_out, 0.0003 - max_inward_push, places=9)

    # 4. Smooth taper to current_curvature when receding strongly (d_dot >= 0.08 m/s).
    guided_unwind, _ = compute_lka_guided_curvature(
      current_curvature=-0.0004,
      anchor_curvature=-0.0004,
      lka_desired_curvature=-0.0004,
      d_dot=0.10,
      v_ego=v_ego,
      nudge_right=False,
    )
    self.assertAlmostEqual(guided_unwind, -0.0004, places=9)

  def test_driver_steering_resets_30s_window_before_warning(self):
    lka = LaneKeepController()
    drifting = _model(left_y=-0.99, right_y=2.6)
    centered = _model(left_y=-1.8, right_y=1.8)

    # Accumulate 3.5s of active nudging (< 5.0s warning threshold): no alert yet.
    for _ in range(350):
      lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.NONE)

    # Return to lane center and apply genuine driver steering (>= 85 CAN for 0.12s): resets 30s rolling window cleanly.
    for _ in range(30):
      lka.update(centered, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    for _ in range(15):
      lka.update(centered, v_ego=18.0, steering_angle_deg=0.0, steering_torque=95.0)
    self.assertEqual(lka._window_active_dt, 0.0)

    # Another 3.5s of nudging within the same 30s span (7.0s total) stays below 5.0s warning.
    for _ in range(350):
      lka.update(drifting, v_ego=18.0, steering_angle_deg=0.0, steering_torque=0.0)
    self.assertEqual(lka.alert, LaneKeepAlert.NONE)

    # Brief 65 CAN EPS-inertia inward spike (0.06s) during active right-line nudge must NOT clear hands-off tracking
    # (accumulates ~5.0s + 0.37s pause = ~5.4s, well before a 2.5s + 5.0s = 7.5s post-reset warning).
    lka_inertia = LaneKeepController()
    drifting_right = _model(left_y=-2.6, right_y=0.90)
    for tick in range(560):
      tq = 65.0 if 250 <= tick < 256 else 0.0
      lka_inertia.update(drifting_right, v_ego=18.0, steering_angle_deg=0.0, steering_torque=tq)
    self.assertEqual(lka_inertia.alert, LaneKeepAlert.KEEP_HANDS_ON_WHEEL)

  def test_controlsd_lka_branch_binds_required_locals(self):
    import ast
    from pathlib import Path

    controlsd_path = Path(__file__).resolve().parents[3] / "selfdrive" / "controls" / "controlsd.py"
    tree = ast.parse(controlsd_path.read_text(encoding="utf-8"), filename=str(controlsd_path))
    controls_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Controls")
    state_control = next(n for n in controls_cls.body if isinstance(n, ast.FunctionDef) and n.name == "state_control")

    lka_ifs = [
      n for n in state_control.body
      if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "lka_actuating"
    ]
    self.assertGreaterEqual(len(lka_ifs), 2)
    lka_if = lka_ifs[-1]
    if_idx = state_control.body.index(lka_if)

    def _assigned_locals(stmts):
      names = set()
      for stmt in stmts:
        for node in ast.walk(stmt):
          if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
      return names

    def _loaded_locals(stmts):
      names = set()
      for stmt in stmts:
        for node in ast.walk(stmt):
          if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
      return names

    prior_assigned = _assigned_locals(state_control.body[:if_idx]) | {arg.arg for arg in state_control.args.args}
    body_assigned = _assigned_locals(lka_if.body)
    else_assigned = _assigned_locals(lka_if.orelse)
    post_loaded = _loaded_locals(state_control.body[if_idx + 1:])

    for required in ("factor", "new_desired_curvature", "assist_ready", "turn_admitted"):
      self.assertIn(required, body_assigned, f"{required} missing from `if lka_actuating:` branch")
      self.assertIn(required, else_assigned, f"{required} missing from `else:` branch")

    # lane_base_curvature is only read when self.last_lane_centering_result is set (inside `else:`).
    only_else = (else_assigned - body_assigned - prior_assigned - {"lane_base_curvature"}) & post_loaded
    self.assertEqual(only_else, set(), f"Locals bound only in `else:` and read later: {sorted(only_else)}")

  def test_confirmed_recede_releases_promptly_without_chatter_and_catches_redrift(self):
    lka = LaneKeepController()
    # 1. Start and hold an active left-line departure nudge (d_left = -0.05m).
    for tick in range(30):
      lka.update(_model(left_y=-0.91, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0,
                 steering_torque=0.0, model_updated=(tick % 5 == 0))
    self.assertTrue(lka.active)
    self.assertGreater(lka.desired_curvature, 0.0)
    prev_window_active = lka._window_active_dt
    self.assertGreater(prev_window_active, 0.0)

    # 2. Vehicle steers/drifts cleanly back into the lane (+0.01m per 50ms = +0.20 m/s) while still inside
    #    d_thresh = +0.07m: confirmed recede should drop active assist promptly, keep amber HUD hold, and NOT clear 30s window.
    released_step = None
    left_y = -0.91
    for step in range(10):
      left_y -= 0.01
      for sub in range(5):
        lka.update(_model(left_y=left_y, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0,
                   steering_torque=0.0, model_updated=(sub == 0))
        if released_step is None and not lka.active:
          released_step = step * 5 + sub
    self.assertIsNotNone(released_step)
    self.assertEqual(lka.desired_curvature, 0.0)
    self.assertEqual(lka.left_line_state, 3)
    self.assertGreater(lka._window_active_dt, 0.0)

    # 3. Brief flat/slow crawl still near the line (d_left ~ +0.03m < +0.07m) immediately after release must not chatter back on.
    for tick in range(25):
      lka.update(_model(left_y=-0.99, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0,
                 steering_torque=0.0, model_updated=(tick % 5 == 0))
      self.assertFalse(lka.active)

    # 4. Genuine outward re-drift (-0.012m per 50ms = -0.24 m/s) breaks the recede cooldown early and re-engages LKA.
    reengaged = False
    left_y = -0.99
    for _ in range(10):
      left_y += 0.012
      for sub in range(5):
        lka.update(_model(left_y=left_y, right_y=2.6), v_ego=18.0, steering_angle_deg=0.0,
                   steering_torque=0.0, model_updated=(sub == 0))
        if lka.active:
          reengaged = True
    self.assertTrue(reengaged)
    self.assertGreater(lka.desired_curvature, 0.0)


if __name__ == "__main__":
  unittest.main()
