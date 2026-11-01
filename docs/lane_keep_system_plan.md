# Architecture & Implementation Plan: 2025 Subaru Legacy Lane Keep Assist (LKA)

## 1. Executive Summary & Design Vision

This system implements a comma-safety-compliant **Lane Keep Assist (LKA)** system exclusively gated to the **2025 Subaru Legacy (`CAR.SUBARU_LEGACY_2025`)**, operating alongside continuous **Lane Centering Assist (MADS / AOL)** as two orthogonal lateral modes gated by the vehicle's hardware **Main Cruise (`Cruise_On`)** switch.

### Core Behaviors
- **2025 Subaru Legacy Gating**: All LKA runtime control, cluster P/R/N ready-wheel feedback, CVT paddle-shifter manual-gear (`1..8`) allowances, on-device settings, and Galaxy controls are strictly gated to `CAR.SUBARU_LEGACY_2025`.
- **Hardware Master Safety Gate (`Cruise_On`)**: All lateral steering authority (both LKA nudges and MADS/AOL continuous centering) is strictly gated by `ES_DashStatus.Cruise_On == 1` (`0x321`) inside STM32 Panda safety (`acc_main_on`) and software.
  - When `Cruise_On == 1` in **Park / Reverse / Neutral**, the cluster shows the **White "Ready" Steering Wheel (`LKAS_Dash_State = 1`)** with lane lines hidden (`0`) and steering actuation strictly disabled (`LKAS_Request = 0`).
  - In **Drive** and **CVT Manual / Paddle-Shift Mode (`GearShifter.manumatic`, gears `1..8`)**, both LKA-Only and MADS/AOL operate seamlessly without dropping out.
- **Orthogonal Longitudinal x Lateral Mode Selection**:
  - **Main Cruise Button**: Arms or disarms the lateral system (`Cruise_On: 0 <-> 1`). If ACC is currently engaged, the first press cancels ACC while keeping the chosen lateral mode active; a second press turns `Cruise_On` off and disarms all lateral control.
  - **Lane Centering Button (`ButtonType.lkas`)**: Toggles cleanly between **LKA-Only** (White wheel `1`) and **MADS / Continuous Centering** (Green wheel `2`) whether ACC is OFF or ON.
  - **Persistent Last-Mode Memory (`LastLateralModeCentering`)**: Remembers which lateral mode was used last across ignition cycles and restores it when Main Cruise is armed.
- **Painted-Lane-Line-Only Virtual Rumble Strip**:
  - 100% manual steering while safely inside painted lane lines.
  - Standard **`~3 in` (`0.07 m`)** clearance margin (`0.75 s` TTC inside `0.10 m`), immutable entry-curvature boundary-parallel alignment (preventing multi-wave curvature stacking), immediate inward driver recovery release, and **`>= 1.0 s` (`1.05 s`)** minimum solid yellow cluster line hold without threshold flicker.
- **Rolling 30-Second Anti-Lazy Timeout**:
  - **`5.0 s`** of cumulative non-overridden LKA correction within the last **`30.0 s`** triggers `"Keep Hands on Steering Wheel"` (`LaneKeepAlert.KEEP_HANDS_ON_WHEEL`).
  - **`15.0 s`** (`5.0 s` trigger + `10.0 s` hands-off grace timer, or `15.0 s` active correction in `30.0 s`) cuts off LKA actuation (`_lockout = True`) and triggers the elevated `"Lane Keep Assist Timed Out"` alert until the driver applies hands-on steering torque.
- **Persistent Parameters**:
  - `LaneKeepAssist` (`BOOL`, default `False`): Master enable toggle (visible only on `SUBARU_LEGACY_2025`).
  - `LaneKeepMinActivationSpeed` (`FLOAT`, default `25.0`, range `15..45 mph` with `4 mph` hysteresis).
  - `LastLateralModeCentering` (`BOOL`, default `True`): Cross-drive memory of MADS/Centering vs LKA-Only.
