import json
from pathlib import Path


_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "assets/device_settings_layout.json"


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

  assert mode["ui_type"] == "dropdown"
  assert mode["parent_key"] == "LaneKeepAssist"
  assert len(mode["options"]) == 2

  assert sensitivity["ui_type"] == "dropdown"
  assert sensitivity["parent_key"] == "LaneKeepAssist"
  assert len(sensitivity["options"]) == 3

  assert min_speed["ui_type"] == "numeric"
  assert min_speed["parent_key"] == "LaneKeepAssist"
  assert min_speed["min"] == 15.0
  assert min_speed["max"] == 45.0
  assert min_speed["step"] == 1.0
