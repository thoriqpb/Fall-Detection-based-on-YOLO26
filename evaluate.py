"""
evaluate.py — Evaluasi kuantitatif sistem deteksi jatuh pada dataset publik.

Pendekatan: evaluasi PER-KLIP (paling robust & cepat untuk laporan).
  - Setiap klip diberi label tanah-kebenaran (ground truth): 1 = jatuh, 0 = normal.
  - Sistem dijalankan atas seluruh frame klip; prediksi = 1 bila alarm PERNAH
    menyala selama klip, 0 bila tidak.
  - Dihitung confusion matrix, precision, recall, F1, accuracy (kelas positif = jatuh).

Struktur dataset yang didukung (label dari nama folder):
    <dataset>/
        fall/      <- klip JATUH (boleh .mp4 ATAU subfolder berisi frame .png/.jpg)
        normal/    <- klip NORMAL/ADL (sinonim: adl/)
Contoh klip:
    fall/fall-01.mp4                 (video)
    fall/fall-01/0001.png ...        (folder berisi urutan frame, format URFD)

Pemakaian:
    python evaluate.py --dataset data/urfd
    python evaluate.py --dataset data/urfd --fps 30 --limit 10
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import config
from src.detector import PoseDetector
from src.fall_logic import FallTracker

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FALL_DIRS = {"fall", "falls", "jatuh"}
NORMAL_DIRS = {"normal", "adl", "notfall", "aman"}


# ---- iterasi frame per klip ---------------------------------------------
def iter_frames_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS_FALLBACK
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def iter_frames_folder(path):
    files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for p in files:
        img = cv2.imread(str(p))
        if img is not None:
            yield img


def discover_clips(root):
    """Kembalikan daftar (clip_id, label, kind, path)."""
    root = Path(root)
    clips = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if name in FALL_DIRS:
            label = 1
        elif name in NORMAL_DIRS:
            label = 0
        else:
            continue
        for item in sorted(sub.iterdir()):
            if item.is_file() and item.suffix.lower() == ".mp4":
                clips.append((item.stem, label, "video", item))
            elif item.is_dir():
                # folder berisi frame
                if any(p.suffix.lower() in IMAGE_EXTS for p in item.iterdir()):
                    clips.append((item.name, label, "folder", item))
    return clips


# ---- evaluasi satu klip --------------------------------------------------
def evaluate_clip(detector, kind, path, fps):
    """Jalankan sistem atas satu klip. Return (predicted, n_frames, t_alarm_pertama)."""
    tracker = FallTracker()
    frames = iter_frames_video(path) if kind == "video" else iter_frames_folder(path)

    predicted = 0
    first_alarm_frame = None
    n = 0
    base = 1000.0
    for idx, frame in enumerate(frames):
        n = idx + 1
        now = base + idx / fps            # jam sintetis agar cue temporal bermakna
        frame_h = frame.shape[0]
        for det in detector.track(frame, conf=config.CONF_THRESHOLD):
            res = tracker.update(det, frame_h, now)
            if res.is_alarm and predicted == 0:
                predicted = 1
                first_alarm_frame = idx
        tracker.cleanup(now)
    return predicted, n, first_alarm_frame


# ---- metrik & plot -------------------------------------------------------
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_score, recall_score, f1_score)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n_clips": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }


def plot_confusion(cm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Normal", "Jatuh"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Prediksi Sistem")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix Deteksi Jatuh")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=14)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(metrics, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["accuracy", "precision", "recall", "f1"]
    vals = [metrics[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(keys, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Metrik Evaluasi (kelas positif = Jatuh)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluasi kuantitatif deteksi jatuh")
    ap.add_argument("--dataset", required=True, help="folder dataset (berisi fall/ & normal/)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="asumsi FPS untuk klip berupa folder-frame")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah klip (0=semua)")
    args = ap.parse_args()

    clips = discover_clips(args.dataset)
    if not clips:
        raise SystemExit(
            f"[ERROR] Tidak menemukan klip di {args.dataset}. "
            "Pastikan ada subfolder fall/ dan normal/."
        )
    if args.limit:
        # ambil seimbang dari tiap kelas
        falls = [c for c in clips if c[1] == 1][: args.limit]
        norms = [c for c in clips if c[1] == 0][: args.limit]
        clips = falls + norms

    print(f"[INFO] {len(clips)} klip ditemukan. Memuat model...")
    detector = PoseDetector()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    y_true, y_pred = [], []
    t0 = time.time()
    for i, (clip_id, label, kind, path) in enumerate(clips, 1):
        pred, n, first = evaluate_clip(detector, kind, path, args.fps)
        y_true.append(label)
        y_pred.append(pred)
        ok = "OK " if pred == label else "XX "
        print(f"  [{i}/{len(clips)}] {ok} {clip_id:<24} gt={label} pred={pred} "
              f"({n} frame, alarm@{first})")
        rows.append({"clip": clip_id, "gt": label, "pred": pred,
                     "n_frames": n, "first_alarm_frame": first})

    metrics = compute_metrics(y_true, y_pred)
    metrics["eval_seconds"] = round(time.time() - t0, 1)
    metrics["model"] = detector.model_name

    # simpan hasil
    (config.RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    import csv as _csv
    with open(config.RESULTS_DIR / "per_clip_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["clip", "gt", "pred", "n_frames",
                                           "first_alarm_frame"])
        w.writeheader()
        w.writerows(rows)
    plot_confusion(metrics["confusion_matrix"], config.RESULTS_DIR / "confusion_matrix.png")
    plot_metrics_bar(metrics, config.RESULTS_DIR / "metrics_bar.png")

    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 48)
    print(f"  Klip dievaluasi : {metrics['n_clips']}")
    print(f"  Accuracy        : {metrics['accuracy']:.3f}")
    print(f"  Precision       : {metrics['precision']:.3f}")
    print(f"  Recall          : {metrics['recall']:.3f}")
    print(f"  F1-score        : {metrics['f1']:.3f}")
    print(f"  Confusion [[TN,FP],[FN,TP]] = {cm}")
    print(f"\n  Hasil & plot disimpan di: {config.RESULTS_DIR}")
    print("=" * 48)


if __name__ == "__main__":
    main()
