"""
config.py — Konfigurasi terpusat Sistem Deteksi Jatuh Lansia.

Semua parameter yang bisa di-tuning dikumpulkan di sini agar tidak ada lagi
"magic number" yang tersebar di banyak file. Threshold deteksi nantinya
dikalibrasi dari hasil evaluasi dataset (lihat evaluate.py), bukan ditebak manual.
"""

from pathlib import Path

# ==========================================================================
# PATH PROYEK
# ==========================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
ASSETS_DIR = BASE_DIR / "assets"

# ==========================================================================
# MODEL YOLO POSE
# ==========================================================================
# Loader (src/detector.py) akan mencoba kandidat ini BERURUTAN dan memakai
# yang pertama berhasil dimuat. Dengan begitu:
#   - ultralytics >= 8.4.x  -> otomatis pakai YOLO26 (sesuai branding proyek)
#   - ultralytics >= 8.3.x  -> jatuh ke YOLO11
#   - ultralytics 8.2.x     -> jatuh ke YOLOv8 (baseline yang pasti tersedia)
MODEL_CANDIDATES = [
    "yolo26n-pose.pt",
    "yolo11n-pose.pt",
    str(MODELS_DIR / "yolov8n-pose.pt"),
    "yolov8n-pose.pt",
]

# Inference
CONF_THRESHOLD = 0.5      # ambang confidence deteksi orang
IOU_THRESHOLD = 0.5       # NMS IoU
KPT_CONF_THRESHOLD = 0.5  # ambang confidence per-keypoint (pengganti cek '== 0.0')
DEVICE = "cpu"            # Paksa CPU — hindari mismatch torchvision::nms CUDA.
                          # Ganti ke None untuk auto-detect GPU bila torch+torchvision
                          # sudah diinstall dalam versi CUDA yang kompatibel.

# Tracking (multi-orang) — ByteTrack bawaan ultralytics
TRACKER_CFG = "bytetrack.yaml"

# ==========================================================================
# KEYPOINT INDEX (format COCO 17 titik)
# ==========================================================================
KP_NOSE = 0
KP_SHOULDER_L = 5
KP_SHOULDER_R = 6
KP_HIP_L = 11
KP_HIP_R = 12
KP_KNEE_L = 13
KP_KNEE_R = 14
KP_ANKLE_L = 15
KP_ANKLE_R = 16

# ==========================================================================
# PARAMETER LOGIKA DETEKSI JATUH (state machine, src/fall_logic.py)
# ==========================================================================
# --- Cue postur "rebah" (low posture) ---
ASPECT_RATIO_THRESHOLD = 0.8   # height/width bbox < nilai ini => cenderung horizontal
TORSO_ANGLE_THRESHOLD = 50.0   # sudut batang tubuh thd vertikal (derajat); > ini => rebah
                               # 0deg = tegak lurus berdiri, 90deg = tubuh horizontal

# --- Cue temporal "jatuh cepat" (rapid drop) ---
# Kecepatan turun pusat massa tubuh, dinormalisasi terhadap TINGGI FRAME
# (skala-invarian thd resolusi), dalam satuan (fraksi tinggi frame) per detik.
# Jatuh nyata menempuh sebagian besar tinggi frame dengan cepat, jauh lebih
# cepat daripada menunduk/duduk perlahan.
VELOCITY_DROP_THRESHOLD = 0.6
HISTORY_SECONDS = 1.0          # panjang window riwayat posisi untuk hitung kecepatan

# --- Konfirmasi & penahanan status ---
FALLEN_CONFIRM_SECONDS = 0.5   # postur rebah harus bertahan selama ini -> status FALLEN
ALARM_CONFIRM_SECONDS = 1.5    # FALLEN bertahan selama ini -> picu ALARM (hindari duduk sesaat)
HOLD_DURATION = 5.0            # status bahaya ditahan selama ini meski frame sempat 'aman'
RECOVER_SECONDS = 1.0          # postur tegak bertahan selama ini -> kembali NORMAL

# ==========================================================================
# NOTIFIKASI (src/notifier.py)
# ==========================================================================
ENABLE_SOUND_ALARM = True
ENABLE_TELEGRAM = False        # set True + isi token/chat id (via env var) untuk aktifkan
TELEGRAM_BOT_TOKEN_ENV = "FALLDET_TG_TOKEN"   # ambil dari environment variable
TELEGRAM_CHAT_ID_ENV = "FALLDET_TG_CHAT"
NOTIFY_COOLDOWN_SECONDS = 30   # jeda minimal antar notifikasi untuk orang yang sama

# ==========================================================================
# OUTPUT & LOGGING
# ==========================================================================
EVENT_LOG_PATH = OUTPUTS_DIR / "event_log.csv"
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = OUTPUTS_DIR / "hasil_deteksi.mp4"
DEFAULT_FPS_FALLBACK = 30      # dipakai bila sumber (webcam) tidak melaporkan FPS
