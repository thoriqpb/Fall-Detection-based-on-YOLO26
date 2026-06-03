"""
visualizer.py — Anotasi visual & HUD (Heads-Up Display) untuk demo live.

Menggambar di atas frame:
  - Bounding box per orang, warnanya mengikuti status (hijau/oranye/merah).
  - Skeleton ringkas + garis batang tubuh (memvisualkan cue torso angle).
  - Label ID + status + durasi jatuh.
  - Panel HUD: FPS, jumlah orang, jumlah dalam bahaya, jam, log kejadian terakhir.

Sengaja memakai OpenCV murni (bukan GUI eksternal) supaya andal saat demo live.
"""

import time

import cv2

import config
from src.fall_logic import FallState

# Warna BGR per status
STATE_COLORS = {
    FallState.NORMAL: (0, 200, 0),     # hijau
    FallState.FALLEN: (0, 165, 255),   # oranye
    FallState.ALARM: (0, 0, 255),      # merah
}

# Pasangan keypoint untuk skeleton ringkas (subset COCO yang relevan)
SKELETON_EDGES = [
    (config.KP_SHOULDER_L, config.KP_SHOULDER_R),
    (config.KP_SHOULDER_L, config.KP_HIP_L),
    (config.KP_SHOULDER_R, config.KP_HIP_R),
    (config.KP_HIP_L, config.KP_HIP_R),
    (config.KP_HIP_L, config.KP_KNEE_L),
    (config.KP_HIP_R, config.KP_KNEE_R),
    (config.KP_KNEE_L, config.KP_ANKLE_L),
    (config.KP_KNEE_R, config.KP_ANKLE_R),
]


def _put_text(img, text, org, color, scale=0.6, thick=2, bg=True):
    """Teks dengan latar gelap semi agar terbaca di atas citra apa pun."""
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_detection(frame, detection, result):
    """Gambar bbox, skeleton, garis batang tubuh, dan label untuk satu orang."""
    color = STATE_COLORS[result.state]
    x1, y1, x2, y2 = (int(v) for v in detection.bbox)
    thick = 3 if result.is_alarm else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    # skeleton ringkas
    for a, b in SKELETON_EDGES:
        pa = detection.keypoint(a)
        pb = detection.keypoint(b)
        if pa is not None and pb is not None:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     (255, 255, 255), 1, cv2.LINE_AA)

    # garis batang tubuh (hip_mid -> shoulder_mid) menyorot cue sudut
    sh_l, sh_r = detection.keypoint(config.KP_SHOULDER_L), detection.keypoint(config.KP_SHOULDER_R)
    hp_l, hp_r = detection.keypoint(config.KP_HIP_L), detection.keypoint(config.KP_HIP_R)
    if all(p is not None for p in (sh_l, sh_r, hp_l, hp_r)):
        sh = ((sh_l[0] + sh_r[0]) / 2, (sh_l[1] + sh_r[1]) / 2)
        hp = ((hp_l[0] + hp_r[0]) / 2, (hp_l[1] + hp_r[1]) / 2)
        cv2.line(frame, (int(hp[0]), int(hp[1])), (int(sh[0]), int(sh[1])), color, 2, cv2.LINE_AA)

    # label
    tid = detection.track_id
    label = f"ID{tid} {result.state.value}"
    if result.cues.torso_angle is not None:
        label += f"  {result.cues.torso_angle:.0f}deg"
    _put_text(frame, label, (x1, max(y1 - 8, 16)), color, scale=0.6, thick=2)

    # durasi jatuh bila relevan
    if result.fallen_since is not None and result.state != FallState.NORMAL:
        dur = time.time() - result.fallen_since
        _put_text(frame, f"jatuh {dur:.1f}s", (x1, y2 + 18), color, scale=0.55, thick=1)
    return frame


def draw_hud(frame, fps, num_people, num_alarm, events):
    """Panel info di bagian atas frame + log kejadian terakhir."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    clock = time.strftime("%H:%M:%S")
    status = "BAHAYA" if num_alarm > 0 else "AMAN"
    status_color = (0, 0, 255) if num_alarm > 0 else (0, 220, 0)
    _put_text(frame, f"FPS:{fps:4.1f}", (8, 24), (255, 255, 255), 0.6, 2, bg=False)
    _put_text(frame, f"Orang:{num_people}", (120, 24), (255, 255, 255), 0.6, 2, bg=False)
    _put_text(frame, f"Sistem: {status}", (250, 24), status_color, 0.6, 2, bg=False)
    _put_text(frame, clock, (w - 90, 24), (255, 255, 255), 0.6, 2, bg=False)

    # banner besar saat ada bahaya
    if num_alarm > 0:
        _put_text(frame, "!!! TERDETEKSI JATUH !!!", (int(w / 2) - 180, 64),
                  (0, 0, 255), 0.9, 3)

    # log kejadian terakhir (maksimal 4 baris di kiri-bawah)
    for i, ev in enumerate(events[-4:]):
        _put_text(frame, ev, (8, h - 8 - i * 22), (0, 200, 255), 0.5, 1)
    return frame
