"""
fall_logic.py — Logika deteksi jatuh berbasis STATE MACHINE per-orang.

Perbaikan utama dibanding prototipe minggu-1 (testFall.py):

1. MULTI-CUE, bukan sinyal tunggal. Menggabungkan:
   - Aspect ratio bounding box (tinggi/lebar).
   - Sudut batang tubuh (torso angle) thd sumbu vertikal -> TIDAK bergantung
     pada hidung, sehingga orang tengkurap/menyamping tetap terdeteksi
     (mengatasi FALSE NEGATIVE pada laporan minggu-1).
   - Kecepatan turun pusat tubuh (cue temporal) -> membedakan "jatuh" dari
     "menunduk/duduk perlahan" (mengatasi FALSE POSITIVE).

2. STATE MACHINE dengan konfirmasi durasi, bukan keputusan 1-frame:
       NORMAL --(rebah cepat / rebah bertahan)--> FALLEN --(bertahan)--> ALARM
       FALLEN/ALARM --(tegak bertahan)--> NORMAL
   Postur rebah harus BERTAHAN beberapa saat sebelum memicu alarm, sehingga
   gerakan sesaat (membungkuk ambil barang, duduk lantai sebentar) tidak
   langsung memicu alarm.

3. Setiap orang (track_id) punya state machine sendiri -> mendukung multi-orang.
   Tidak ada lagi variabel global seperti `last_fall_time`.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

import config


class FallState(Enum):
    NORMAL = "AMAN"
    FALLEN = "TERJATUH"     # postur rebah terkonfirmasi, belum cukup lama untuk alarm
    ALARM = "JATUH! (BAHAYA)"


@dataclass
class FallCues:
    """Hasil perhitungan cue mentah untuk satu frame (berguna untuk debug/HUD)."""
    aspect_ratio: float = 0.0
    torso_angle: Optional[float] = None     # derajat; None bila keypoint tak cukup
    drop_velocity: float = 0.0              # fraksi tinggi-frame per detik (+ = turun)
    is_low_posture: bool = False
    is_rapid_drop: bool = False


@dataclass
class FallResult:
    """Output evaluasi satu orang pada satu frame."""
    state: FallState
    is_alarm: bool
    cues: FallCues
    fallen_since: Optional[float] = None    # timestamp awal rebah (untuk durasi)


def _midpoint(p1: Optional[np.ndarray], p2: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return (p1 + p2) / 2.0


def compute_torso_angle(detection) -> Optional[float]:
    """
    Sudut batang tubuh terhadap sumbu VERTIKAL, dalam derajat.
      ~0 derajat   = tubuh tegak (berdiri/duduk tegak)
      ~90 derajat  = tubuh horizontal (rebah)
    Vektor dihitung dari titik tengah pinggul -> titik tengah bahu.
    Mengembalikan None bila keypoint bahu/pinggul tidak cukup terdeteksi.
    """
    sh = _midpoint(
        detection.keypoint(config.KP_SHOULDER_L),
        detection.keypoint(config.KP_SHOULDER_R),
    )
    hip = _midpoint(
        detection.keypoint(config.KP_HIP_L),
        detection.keypoint(config.KP_HIP_R),
    )
    if sh is None or hip is None:
        return None

    dx = float(sh[0] - hip[0])
    dy = float(sh[1] - hip[1])   # ingat: y membesar ke bawah
    if dx == 0 and dy == 0:
        return None

    # Sudut vektor batang tubuh terhadap vertikal (atas).
    # Vektor "tegak ideal" menunjuk ke atas = (0, -1).
    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle_from_vertical


class FallStateMachine:
    """State machine deteksi jatuh untuk SATU orang (satu track_id)."""

    def __init__(self):
        self.state = FallState.NORMAL
        self.history = deque()              # (timestamp, centroid_y_ternormalisasi)
        self.low_posture_since: Optional[float] = None
        self.upright_since: Optional[float] = None
        self.fallen_since: Optional[float] = None
        self.hold_until: float = 0.0        # penahanan status bahaya
        self.last_seen: float = time.time()

    # ---- perhitungan cue -------------------------------------------------
    def _update_velocity(self, detection, frame_h: int, now: float) -> float:
        """Kecepatan turun pusat tubuh, fraksi tinggi-frame per detik."""
        cy = (detection.bbox[1] + detection.bbox[3]) / 2.0
        cy_norm = cy / max(frame_h, 1)
        self.history.append((now, cy_norm))

        # buang riwayat lebih tua dari window
        cutoff = now - config.HISTORY_SECONDS
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        if len(self.history) < 2:
            return 0.0
        t0, y0 = self.history[0]
        t1, y1 = self.history[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return 0.0
        return (y1 - y0) / dt   # positif = pusat tubuh bergerak turun

    def _compute_cues(self, detection, frame_h: int, now: float) -> FallCues:
        width = max(detection.width, 1.0)
        aspect_ratio = detection.height / width
        torso_angle = compute_torso_angle(detection)
        drop_velocity = self._update_velocity(detection, frame_h, now)

        # Postur rebah: cukup salah satu sinyal kuat terpenuhi.
        low_by_ratio = aspect_ratio < config.ASPECT_RATIO_THRESHOLD
        low_by_angle = (
            torso_angle is not None and torso_angle > config.TORSO_ANGLE_THRESHOLD
        )
        is_low = low_by_ratio or low_by_angle
        is_rapid_drop = drop_velocity > config.VELOCITY_DROP_THRESHOLD

        return FallCues(
            aspect_ratio=aspect_ratio,
            torso_angle=torso_angle,
            drop_velocity=drop_velocity,
            is_low_posture=is_low,
            is_rapid_drop=is_rapid_drop,
        )

    # ---- transisi state --------------------------------------------------
    def update(self, detection, frame_h: int, now: Optional[float] = None) -> FallResult:
        now = now if now is not None else time.time()
        self.last_seen = now
        cues = self._compute_cues(detection, frame_h, now)

        # lacak berapa lama postur rebah / tegak bertahan
        if cues.is_low_posture:
            if self.low_posture_since is None:
                self.low_posture_since = now
            self.upright_since = None
        else:
            self.low_posture_since = None
            if self.upright_since is None:
                self.upright_since = now

        low_duration = (now - self.low_posture_since) if self.low_posture_since else 0.0
        upright_duration = (now - self.upright_since) if self.upright_since else 0.0

        # Jatuh cepat mempersingkat waktu konfirmasi (rebah + barusan menukik).
        fast_confirm = cues.is_rapid_drop and cues.is_low_posture

        # --- logika transisi ---
        if self.state == FallState.NORMAL:
            if fast_confirm or low_duration >= config.FALLEN_CONFIRM_SECONDS:
                self.state = FallState.FALLEN
                self.fallen_since = self.low_posture_since or now

        elif self.state == FallState.FALLEN:
            fallen_duration = (now - self.fallen_since) if self.fallen_since else 0.0
            if fallen_duration >= config.ALARM_CONFIRM_SECONDS:
                self.state = FallState.ALARM
                self.hold_until = now + config.HOLD_DURATION
            elif upright_duration >= config.RECOVER_SECONDS:
                self._reset_to_normal()

        elif self.state == FallState.ALARM:
            self.hold_until = max(self.hold_until, now + config.HOLD_DURATION) \
                if cues.is_low_posture else self.hold_until
            # keluar dari alarm hanya bila sudah tegak cukup lama DAN masa hold habis
            if upright_duration >= config.RECOVER_SECONDS and now >= self.hold_until:
                self._reset_to_normal()

        is_alarm = self.state == FallState.ALARM
        return FallResult(
            state=self.state,
            is_alarm=is_alarm,
            cues=cues,
            fallen_since=self.fallen_since,
        )

    def _reset_to_normal(self):
        self.state = FallState.NORMAL
        self.fallen_since = None
        self.low_posture_since = None


class FallTracker:
    """Mengelola satu FallStateMachine per track_id, plus pembersihan ID basi."""

    def __init__(self, stale_seconds: float = 3.0):
        self.machines = {}
        self.stale_seconds = stale_seconds

    def update(self, detection, frame_h: int, now: Optional[float] = None) -> FallResult:
        now = now if now is not None else time.time()
        tid = detection.track_id
        if tid not in self.machines:
            self.machines[tid] = FallStateMachine()
        return self.machines[tid].update(detection, frame_h, now)

    def cleanup(self, now: Optional[float] = None):
        """Hapus state orang yang sudah lama tak terlihat (keluar frame)."""
        now = now if now is not None else time.time()
        stale = [
            tid for tid, m in self.machines.items()
            if now - m.last_seen > self.stale_seconds
        ]
        for tid in stale:
            del self.machines[tid]
