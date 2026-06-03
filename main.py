"""
main.py — Entry point Sistem Deteksi Jatuh Lansia (demo live / video).

Menggantikan prototipe testFall.py dengan pipeline modular:
    sumber video -> PoseDetector (deteksi+tracking) -> FallTracker (state machine
    per-orang) -> Visualizer (anotasi+HUD) -> [notifikasi] -> tampil/simpan.

Contoh pemakaian:
    python main.py --source 0                 # webcam
    python main.py --source video.mp4         # file video
    python main.py --source video.mp4 --save  # sekaligus rekam hasil
    python main.py --source 0 --no-display     # tanpa jendela (mis. headless)
"""

import argparse
import csv
import time

import cv2

import config
from src.detector import PoseDetector
from src.fall_logic import FallTracker, FallState


def parse_args():
    p = argparse.ArgumentParser(description="Sistem Deteksi Jatuh Lansia (YOLO Pose)")
    p.add_argument("--source", default="0",
                   help="0/1.. untuk webcam, atau path ke file video")
    p.add_argument("--save", action="store_true", help="rekam hasil ke outputs/")
    p.add_argument("--no-display", action="store_true", help="jangan tampilkan jendela")
    p.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                   help="ambang confidence deteksi")
    p.add_argument("--no-notify", action="store_true", help="matikan notifikasi")
    return p.parse_args()


def open_source(source):
    """Buka webcam (indeks) atau file video (path)."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Tidak dapat membuka sumber video: {source}")
    return cap


def make_writer(cap):
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or config.DEFAULT_FPS_FALLBACK
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(config.OUTPUT_VIDEO_PATH), fourcc, fps, (w, h))
    print(f"[INFO] Merekam hasil ke: {config.OUTPUT_VIDEO_PATH}")
    return out


class EventLogger:
    """Catat kejadian jatuh ke CSV + simpan baris ringkas untuk HUD."""

    def __init__(self):
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = config.EVENT_LOG_PATH
        self.recent = []
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "waktu", "track_id", "kejadian"])

    def log(self, track_id, kejadian):
        ts = time.time()
        waktu = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{ts:.2f}", waktu, track_id, kejadian])
        line = f"{time.strftime('%H:%M:%S')} ID{track_id} {kejadian}"
        self.recent.append(line)
        print(f"[EVENT] {line}")


def main():
    args = parse_args()

    # lazy import agar --help cepat & test tak butuh torch
    from src import visualizer

    notifier = None
    if not args.no_notify:
        try:
            from src.notifier import Notifier
            notifier = Notifier()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Notifikasi nonaktif: {exc}")

    print("[INFO] Memuat model YOLO Pose...")
    detector = PoseDetector()
    tracker = FallTracker()
    logger = EventLogger()

    cap = open_source(args.source)
    writer = make_writer(cap) if args.save else None
    alarmed_ids = set()   # untuk memicu kejadian/notifikasi sekali per kejadian

    print("[INFO] Sistem berjalan. Tekan 'q' untuk berhenti.")
    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Sumber video selesai.")
                break

            now = time.time()
            frame_h = frame.shape[0]
            detections = detector.track(frame, conf=args.conf)

            num_alarm = 0
            current_alarm_ids = set()
            for det in detections:
                result = tracker.update(det, frame_h, now)
                visualizer.draw_detection(frame, det, result)
                if result.is_alarm:
                    num_alarm += 1
                    current_alarm_ids.add(det.track_id)
                    # kejadian BARU (transisi ke alarm)
                    if det.track_id not in alarmed_ids:
                        logger.log(det.track_id, "JATUH terdeteksi")
                        if notifier:
                            notifier.notify_fall(det.track_id, frame)
            alarmed_ids = current_alarm_ids
            tracker.cleanup(now)

            # FPS (rata-rata bergulir sederhana)
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

            visualizer.draw_hud(frame, fps, len(detections), num_alarm, logger.recent)

            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Sistem Deteksi Jatuh Lansia (YOLO Pose)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Dihentikan pengguna.")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
