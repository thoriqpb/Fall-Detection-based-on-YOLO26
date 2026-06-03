"""
test_fall_logic.py — Uji sintetis state machine deteksi jatuh.

Memvalidasi logika tanpa perlu video nyata, dengan membangun PersonDetection
palsu untuk 3 skenario kritis:
  1. Berdiri terus           -> harus tetap AMAN
  2. Rebah & diam berkelanjutan -> harus naik jadi ALARM
  3. Membungkuk sesaat lalu berdiri -> TIDAK boleh memicu alarm (anti false positive)

Jalankan: python -m tests.test_fall_logic   (dari root proyek)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.detector import PersonDetection
from src.fall_logic import FallStateMachine, FallState

FRAME_H = 480
FRAME_W = 640


def _make_keypoints(shoulder, hip, nose=None):
    """Bangun array keypoint (17,2)+conf(17,) dengan bahu, pinggul, hidung terisi."""
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    def setp(idx, pt):
        xy[idx] = pt
        conf[idx] = 0.9

    # bahu kiri/kanan diberi sedikit offset horizontal di sekitar 'shoulder'
    setp(config.KP_SHOULDER_L, [shoulder[0] - 10, shoulder[1]])
    setp(config.KP_SHOULDER_R, [shoulder[0] + 10, shoulder[1]])
    setp(config.KP_HIP_L, [hip[0] - 10, hip[1]])
    setp(config.KP_HIP_R, [hip[0] + 10, hip[1]])
    if nose is not None:
        setp(config.KP_NOSE, nose)
    return xy, conf


def make_standing(cx=320, top=100):
    """Orang berdiri: bbox tinggi, bahu di atas pinggul (torso vertikal)."""
    h, w = 240, 70
    bbox = (cx - w / 2, top, cx + w / 2, top + h)
    shoulder = [cx, top + 40]
    hip = [cx, top + 130]
    nose = [cx, top + 10]
    kxy, kconf = _make_keypoints(shoulder, hip, nose)
    return PersonDetection(1, bbox, kxy, kconf, 0.9)


def make_lying(cy=400, left=200):
    """Orang rebah: bbox lebar, bahu & pinggul sejajar horizontal (torso ~90 derajat)."""
    h, w = 70, 240
    bbox = (left, cy - h / 2, left + w, cy + h / 2)
    shoulder = [left + 60, cy]
    hip = [left + 150, cy]
    nose = [left + 30, cy]
    kxy, kconf = _make_keypoints(shoulder, hip, nose)
    return PersonDetection(1, bbox, kxy, kconf, 0.9)


def run_scenario(name, frames, dt=0.1):
    """frames: list of (detection, label). Kembalikan state akhir & apakah pernah alarm."""
    sm = FallStateMachine()
    t = 1000.0
    ever_alarm = False
    states = []
    for det, _label in frames:
        res = sm.update(det, FRAME_H, now=t)
        states.append(res.state)
        ever_alarm = ever_alarm or res.is_alarm
        t += dt
    print(f"[{name}] state akhir = {states[-1].name}, pernah ALARM = {ever_alarm}")
    return states[-1], ever_alarm


def test_standing_stays_safe():
    frames = [(make_standing(), "berdiri") for _ in range(50)]
    final, ever_alarm = run_scenario("Berdiri 5 dtk", frames)
    assert final == FallState.NORMAL, "Berdiri seharusnya tetap AMAN"
    assert not ever_alarm, "Berdiri tidak boleh memicu alarm"


def test_lying_triggers_alarm():
    # 0.5 dtk berdiri, lalu rebah & diam 4 dtk
    frames = [(make_standing(), "berdiri") for _ in range(5)]
    frames += [(make_lying(), "rebah") for _ in range(40)]
    final, ever_alarm = run_scenario("Rebah berkelanjutan", frames)
    assert final == FallState.ALARM, "Rebah berkelanjutan seharusnya ALARM"
    assert ever_alarm


def test_brief_bend_no_false_alarm():
    # berdiri, membungkuk SEBENTAR (0.3 dtk, < FALLEN_CONFIRM), lalu berdiri lagi
    frames = [(make_standing(), "berdiri") for _ in range(10)]
    frames += [(make_lying(), "membungkuk") for _ in range(3)]   # 0.3 dtk saja
    frames += [(make_standing(), "berdiri") for _ in range(20)]
    final, ever_alarm = run_scenario("Membungkuk sesaat", frames)
    assert not ever_alarm, "Membungkuk sesaat TIDAK boleh memicu alarm"
    assert final == FallState.NORMAL


if __name__ == "__main__":
    print(f"Param: FALLEN_CONFIRM={config.FALLEN_CONFIRM_SECONDS}s, "
          f"ALARM_CONFIRM={config.ALARM_CONFIRM_SECONDS}s, "
          f"ASPECT<{config.ASPECT_RATIO_THRESHOLD}, ANGLE>{config.TORSO_ANGLE_THRESHOLD}deg\n")
    failures = 0
    for fn in [test_standing_stays_safe, test_lying_triggers_alarm,
               test_brief_bend_no_false_alarm]:
        try:
            fn()
            print(f"  PASS: {fn.__name__}\n")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL: {fn.__name__} -> {e}\n")
    print("=" * 50)
    print("SEMUA TEST LULUS" if failures == 0 else f"{failures} TEST GAGAL")
    sys.exit(1 if failures else 0)
