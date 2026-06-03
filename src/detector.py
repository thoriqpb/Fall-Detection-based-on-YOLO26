"""
detector.py — Wrapper di atas YOLO Pose dari Ultralytics.

Tanggung jawab:
  1. Memuat model secara robust (mencoba beberapa kandidat bobot berurutan).
  2. Menjalankan deteksi + tracking multi-orang (ID persisten via ByteTrack).
  3. Mengubah output mentah Ultralytics menjadi struktur 'PersonDetection'
     yang rapi dan mudah dipakai modul logika (fall_logic) maupun visualizer.

Dengan memisahkan lapisan ini, sisa sistem tidak perlu tahu detail API Ultralytics.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import config

# Catatan: `ultralytics`/`torch` di-import secara LAZY di dalam PoseDetector,
# bukan di level modul. Dengan begitu modul lain (mis. unit test fall_logic)
# bisa memakai dataclass PersonDetection tanpa harus memuat torch yang berat.


@dataclass
class PersonDetection:
    """Satu orang terdeteksi dalam satu frame."""
    track_id: int                 # ID persisten lintas-frame (-1 bila tracking gagal)
    bbox: tuple                   # (x1, y1, x2, y2) dalam piksel
    keypoints_xy: np.ndarray      # shape (17, 2), koordinat piksel
    keypoints_conf: np.ndarray    # shape (17,), confidence tiap keypoint
    box_conf: float               # confidence bounding box

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def keypoint(self, idx: int) -> Optional[np.ndarray]:
        """Kembalikan (x, y) keypoint bila confidence memadai, else None."""
        if self.keypoints_conf[idx] < config.KPT_CONF_THRESHOLD:
            return None
        xy = self.keypoints_xy[idx]
        # Ultralytics memberi (0,0) untuk titik yang tak terdeteksi
        if xy[0] == 0 and xy[1] == 0:
            return None
        return xy


class PoseDetector:
    """Memuat model YOLO Pose dan menjalankan tracking per-frame."""

    def __init__(self, candidates: Optional[List[str]] = None):
        self.candidates = candidates or config.MODEL_CANDIDATES
        self.model_name: Optional[str] = None
        self.model = self._load_model()

    def _load_model(self):
        """Coba tiap kandidat bobot; pakai yang pertama berhasil."""
        from ultralytics import YOLO  # lazy import (memuat torch)

        errors = []
        for cand in self.candidates:
            try:
                model = YOLO(cand)
                self.model_name = cand
                print(f"[detector] Model dimuat: {cand}")
                return model
            except Exception as exc:  # noqa: BLE001 — sengaja tangkap semua agar fallback
                errors.append(f"  - {cand}: {type(exc).__name__}: {exc}")
                continue
        raise RuntimeError(
            "Gagal memuat model YOLO Pose. Kandidat yang dicoba:\n"
            + "\n".join(errors)
            + "\nPastikan koneksi internet aktif atau letakkan bobot di folder models/."
        )

    def track(self, frame, conf=None, iou=None) -> List[PersonDetection]:
        """
        Jalankan deteksi + tracking pada satu frame.
        Mengembalikan daftar PersonDetection (kosong bila tak ada orang).
        """
        results = self.model.track(
            frame,
            persist=True,                       # jaga ID antar pemanggilan
            tracker=config.TRACKER_CFG,
            conf=conf if conf is not None else config.CONF_THRESHOLD,
            iou=iou if iou is not None else config.IOU_THRESHOLD,
            device=config.DEVICE,
            verbose=False,
        )

        detections: List[PersonDetection] = []
        if not results:
            return detections

        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return detections

        kxy = r.keypoints.xy.cpu().numpy()        # (N, 17, 2)
        kconf = (
            r.keypoints.conf.cpu().numpy()        # (N, 17)
            if r.keypoints.conf is not None
            else np.ones(kxy.shape[:2])
        )
        boxes = r.boxes.xyxy.cpu().numpy()        # (N, 4)
        box_conf = r.boxes.conf.cpu().numpy()     # (N,)
        ids = (
            r.boxes.id.cpu().numpy().astype(int)  # (N,)
            if r.boxes.id is not None
            else np.full(len(boxes), -1)
        )

        for i in range(len(boxes)):
            if kxy[i].shape[0] < 17:
                continue  # butuh 17 keypoint COCO lengkap
            detections.append(
                PersonDetection(
                    track_id=int(ids[i]),
                    bbox=tuple(boxes[i].tolist()),
                    keypoints_xy=kxy[i],
                    keypoints_conf=kconf[i],
                    box_conf=float(box_conf[i]),
                )
            )
        return detections

    def plot_skeleton(self, frame):
        """Helper opsional: gambar skeleton bawaan YOLO (untuk debugging)."""
        results = self.model(frame, verbose=False)
        return results[0].plot() if results else frame
