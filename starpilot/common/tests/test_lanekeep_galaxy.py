import json
import re
from collections import Counter
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "assets/device_settings_layout.json"
_PARAMS_KEYS_PATH = _REPO_ROOT / "common/params_keys.h"
_FEASIBLE_PARAMS_PATH = _REPO_ROOT / "tools/StarPilot/feasibleparams.txt"
_THE_GALAXY_PATH = _REPO_ROOT / "starpilot/system/the_galaxy/the_galaxy.py"
_LATERAL_UI_PATH = _REPO_ROOT / "selfdrive/ui/layouts/settings/starpilot/lateral.py"


def _sections():
  layout = json.loads(_LAYOUT_PATH.read_text(encoding="utf-8"))
  return {
    section["name"]: {param["key"]: param for param in section["params"]}
    for section in layout
  }


def test_lanekeep_is_in_lateral_section():
  sections = _sections()
  lateral = sections["Lateral (Steering)"]
  keys = {"LaneKeepAssist", "LaneKeepMode", "LaneKeepSensitivity", "LaneKeepMinActivationSpeed"}

  assert keys <= lateral.keys()


def test_lanekeep_galaxy_controls():
  lateral = _sections()["Lateral (Steering)"]
  lka = lateral["LaneKeepAssist"]
  mode = lateral["LaneKeepMode"]
  sensitivity = lateral["LaneKeepSensitivity"]
  min_speed = lateral["LaneKeepMinActivationSpeed"]

  assert lka["ui_type"] == "toggle"
  assert lka["is_parent_toggle"] is True
  assert lka["data_type"] == "bool"
  assert lka["default"] is False
  assert lka["settings_tier"] == "simple"

  assert mode["ui_type"] == "dropdown"
  assert mode["parent_key"] == "LaneKeepAssist"
  assert len(mode["options"]) == 2
  assert mode["settings_tier"] == "simple"

  assert sensitivity["ui_type"] == "dropdown"
  assert sensitivity["parent_key"] == "LaneKeepAssist"
  assert len(sensitivity["options"]) == 3
  assert sensitivity["settings_tier"] == "simple"

  assert min_speed["ui_type"] == "numeric"
  assert min_speed["parent_key"] == "LaneKeepAssist"
  assert min_speed["min"] == 15.0
  assert min_speed["max"] == 45.0
  assert min_speed["step"] == 1.0
  assert min_speed["settings_tier"] == "simple"


def test_params_keys_no_duplicates_and_lka_persistence_declarations():
  source = _PARAMS_KEYS_PATH.read_text(encoding="utf-8")
  keys = re.findall(r'\{"([A-Za-z0-9_]+)",\s*\{', source)
  counts = Counter(keys)
  duplicates = {k: c for k, c in counts.items() if c > 1}
  assert not duplicates, f"Duplicate keys in params_keys.h: {duplicates}"

  expected_defaults = {
    "LaneKeepAssist": ("BOOL", "0"),
    "LaneKeepMode": ("INT", "2"),
    "LaneKeepSensitivity": ("INT", "1"),
    "LaneKeepMinActivationSpeed": ("FLOAT", "25.0"),
    "LastLateralModeCentering": ("BOOL", "1"),
  }
  for key, (expected_type, expected_default) in expected_defaults.items():
    match = re.search(
      rf'\{{"{re.escape(key)}",\s*\{{PERSISTENT,\s*{expected_type},\s*"{re.escape(expected_default)}"',
      source,
    )
    assert match is not None, f"Missing or mismatched params_keys.h declaration for {key}"

  feasible = {
    line.strip()
    for line in _FEASIBLE_PARAMS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
  }
  assert set(expected_defaults.keys()) <= feasible


def test_the_galaxy_has_no_raw_file_fallback_bandaid():
  galaxy_source = _THE_GALAXY_PATH.read_text(encoding="utf-8")
  assert "_safe_params_put" not in galaxy_source


def test_on_device_lateral_settings_exposes_lanekeep_controls():
  lateral_ui = _LATERAL_UI_PATH.read_text(encoding="utf-8")
  assert '_make_parent("LaneKeepAssist"' in lateral_ui
  assert '"LaneKeepMode"' in lateral_ui
  assert '"LaneKeepSensitivity"' in lateral_ui
  assert '"LaneKeepMinActivationSpeed"' in lateral_ui
