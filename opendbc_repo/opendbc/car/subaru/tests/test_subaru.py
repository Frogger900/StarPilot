from pathlib import Path

from opendbc.car import gen_empty_fingerprint
from opendbc.car.interfaces import get_torque_params
from opendbc.car.structs import CarParams
from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, SubaruFlags, SubaruSafetyFlags


Ecu = CarParams.Ecu
REPO_ROOT = Path(__file__).resolve().parents[5]
SUBARU_CARSTATE = Path(__file__).resolve().parents[1] / "carstate.py"
SERVICES = REPO_ROOT / "cereal/services.py"


class TestSubaruFingerprint:
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          assert len(fw) == fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}"

  def test_crosstrek_2024_platform_is_data_backed_angle_lkas(self):
    config = CAR.SUBARU_CROSSTREK_2024.config

    assert config.car_docs[0].name == "Subaru Crosstrek 2024"
    assert config.flags & SubaruFlags.GLOBAL_GEN2
    assert config.flags & SubaruFlags.LKAS_ANGLE
    assert config.specs.mass == 1529
    assert config.specs.wheelbase == 2.5781
    assert config.specs.steerRatio == 13

  def test_crosstrek_2024_fw_versions_are_present(self):
    fws = FW_VERSIONS[CAR.SUBARU_CROSSTREK_2024]

    assert b'\xa2 $\x18\x03' in fws[(Ecu.abs, 0x7b0, None)]
    assert b'*\xd0\x00\n\x03' in fws[(Ecu.eps, 0x746, None)]
    assert b' \x02\x0e' in fws[(Ecu.fwdCamera, 0x787, None)]
    assert b'\n!\x08\x036\x04!\x08\x01.' in fws[(Ecu.fwdCamera, 0x787, None)]
    assert b'\xe8"ap\x07' in fws[(Ecu.engine, 0x7a2, None)]
    assert b'@\x127cp' in fws[(Ecu.transmission, 0x7a3, None)]

  def test_crosstrek_2025_fw_versions_keep_existing_and_jacob_camera_values(self):
    fws = FW_VERSIONS[CAR.SUBARU_CROSSTREK_2025]

    assert b' \x02\x0e' in fws[(Ecu.fwdCamera, 0x787, None)]
    assert b'\x1d!\x08\x00F\x14!\x08\x00=' in fws[(Ecu.fwdCamera, 0x787, None)]
    assert b'\x1b!\x08\x00D\x11!\x08\x01;' in fws[(Ecu.fwdCamera, 0x787, None)]
    assert b'\xa2 $\x15\x05' in fws[(Ecu.abs, 0x7b0, None)]
    assert b'\xa2 $\x17\x06' in fws[(Ecu.abs, 0x7b0, None)]
    assert b'\xa2 $\x18\x05' in fws[(Ecu.abs, 0x7b0, None)]

  def test_ascent_2023_jacob_camera_fw_version_is_present(self):
    fws = FW_VERSIONS[CAR.SUBARU_ASCENT_2023]

    assert b'\x05!\x08\x1dK\x00\x00\x00\x00\x00' in fws[(Ecu.fwdCamera, 0x787, None)]


class TestSubaruLongitudinalAvailability:
  @staticmethod
  def _params(candidate, *, alpha_long=False, is_release=False, docs=False):
    return CarInterface.get_params(candidate, gen_empty_fingerprint(), [], alpha_long, is_release, docs, None)

  def test_outback_2023_25_does_not_advertise_alpha_long(self):
    for alpha_long, is_release, docs in (
      (False, False, False),
      (True, False, False),
      (True, True, False),
      (True, False, True),
    ):
      cp = self._params(CAR.SUBARU_OUTBACK_2023, alpha_long=alpha_long, is_release=is_release, docs=docs)

      assert not cp.alphaLongitudinalAvailable
      assert not cp.openpilotLongitudinalControl
      assert not (cp.flags & SubaruFlags.DISABLE_EYESIGHT)
      assert not (cp.safetyConfigs[0].safetyParam & SubaruSafetyFlags.LONG)

  def test_forester_2022_24_is_supported_angle_lkas(self):
    cp = self._params(CAR.SUBARU_FORESTER_2022)

    assert cp.flags & SubaruFlags.LKAS_ANGLE
    assert not cp.dashcamOnly
    assert not cp.alphaLongitudinalAvailable
    assert not cp.openpilotLongitudinalControl

  def test_other_lkas_angle_subarus_with_params_remain_blocked(self):
    for candidate in (
      CAR.SUBARU_FORESTER_2022,
      CAR.SUBARU_ASCENT_2023,
      CAR.SUBARU_CROSSTREK_2024,
      CAR.SUBARU_CROSSTREK_2025,
    ):
      cp = self._params(candidate, alpha_long=True)

      assert cp.flags & SubaruFlags.LKAS_ANGLE
      assert not cp.alphaLongitudinalAvailable
      assert not cp.openpilotLongitudinalControl
      assert not (cp.flags & SubaruFlags.DISABLE_EYESIGHT)
      assert not (cp.safetyConfigs[0].safetyParam & SubaruSafetyFlags.LONG)
