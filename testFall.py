import cv2
from ultralytics import YOLO
import time

# ==========================================
# --- KONFIGURASI PENAHANAN STATUS ---
# ==========================================
HOLD_DURATION = 5.0  # Waktu tahan status jatuh (dalam detik)
last_fall_time = 0.0 # Menyimpan timestamp terakhir kali jatuh terdeteksi

# ==========================================
# --- KONFIGURASI PENGGUNA (USER CONFIG) ---
# ==========================================
USE_WEBCAM = 0       # Ubah ke False jika ingin menggunakan file video
SAVE_OUTPUT = 1     # Ubah ke True jika ingin menyimpan hasil deteksi ke file .mp4

VIDEO_PATH = "Kuliah/Pengolahan Citra Digital/video_original.mp4"  # Path ke file video jika USE_WEBCAM = False
OUTPUT_PATH = "Kuliah/Pengolahan Citra Digital/hasil_deteksi_jatuh.mp4" 
MODEL_PATH = 'yolo26n-pose.pt'
# ==========================================

def initialize_video_source():
    """Menginisialisasi input video dari webcam atau file."""
    if USE_WEBCAM:
        video_source = 0
        print("[INFO] Menggunakan input dari Webcam.")
    else:
        video_source = VIDEO_PATH
        print(f"[INFO] Menggunakan input dari file video: {VIDEO_PATH}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka sumber video.")
        exit()
    return cap

def initialize_video_writer(cap):
    """Menginisialisasi objek VideoWriter jika SAVE_OUTPUT True."""
    if not SAVE_OUTPUT:
        print("[INFO] Fitur perekaman TIDAK AKTIF.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        fps = 30 # Default fallback untuk webcam

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"[INFO] Fitur perekaman AKTIF. Hasil akan disimpan di: {OUTPUT_PATH}")
    return out

def check_fall_status(keypoints, box):
    """
    Mengevaluasi logika jatuh berdasarkan keypoints dan bounding box.
    Status JATUH akan ditahan selama beberapa detik meskipun frame saat ini tidak mendeteksi jatuh.
    Mengembalikan nilai boolean True (Jatuh) atau False (Aman).
    """
    global last_fall_time # Gunakan variabel global untuk melacak waktu lintas-frame

    # Ekstrak bounding box
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    width = x2 - x1
    height = y2 - y1

    # Cek apakah sistem sedang dalam masa penahanan status bahaya
    is_currently_holding = (time.time() - last_fall_time) < HOLD_DURATION

    # Jika box tidak valid, kembalikan status hold (jika ada), atau False
    if width <= 0:
        return is_currently_holding, (x1, y1, x2, y2)

    # Ekstrak keypoints (0: Hidung, 11: Pinggul Kiri, 12: Pinggul Kanan)
    nose_y = float(keypoints[0][1])
    hip_left_y = float(keypoints[11][1])
    hip_right_y = float(keypoints[12][1])

    # Pastikan titik-titik krusial tidak terhalang (nilai 0.0)
    if nose_y == 0.0 or hip_left_y == 0.0 or hip_right_y == 0.0:
        return is_currently_holding, (x1, y1, x2, y2)

    # --- LOGIKA MATEMATIKA ---
    ratio = height / width
    avg_hip_y = (hip_left_y + hip_right_y) / 2.0
    
    # Cek murni berdasarkan frame saat ini
    current_frame_is_fallen = (ratio < 0.5) and (nose_y > avg_hip_y)

    # --- LOGIKA PENAHANAN STATUS (STATE HOLDING) ---
    if current_frame_is_fallen:
        # Jika frame ini deteksi jatuh, perbarui waktu terakhir jatuh
        last_fall_time = time.time()
        final_status = True
    else:
        # Jika frame ini mendeteksi aman, cek apakah masih dalam durasi "hold"
        if is_currently_holding:
            final_status = True  # Tahan status bahaya
        else:
            final_status = False # Benar-benar aman

    return final_status, (x1, y1, x2, y2)

def draw_annotations(frame, is_fallen, bbox):
    """Menggambar teks status dan mengubah warna kotak bounding box."""
    x1, y1, x2, y2 = bbox
    
    if is_fallen:
        cv2.putText(frame, "STATUS: JATUH! (BAHAYA)", (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    else:
        cv2.putText(frame, "STATUS: AMAN", (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def main():
    """Fungsi utama (Main Loop) untuk menjalankan keseluruhan sistem."""
    print("[INFO] Memuat model YOLO...")
    model = YOLO(MODEL_PATH)
    
    cap = initialize_video_source()
    out = initialize_video_writer(cap)

    print("--- Sistem Deteksi Jatuh Berjalan ---")
    print("Tekan tombol 'q' pada keyboard untuk berhenti.")

    while True:
        success, frame = cap.read()
        if not success:
            print("[INFO] Video selesai atau frame tidak terbaca.")
            break

        # ==========================================================
        # FITUR BAWAAN YOLO UNTUK MENCEGAH MULTIPLE DETECTION
        # conf=0.6   : Buang deteksi ragu-ragu (< 60% yakin)
        # iou=0.4    : Gabungkan kotak yang tumpang tindih > 40%
        # max_det=1  : Paksa AI hanya mendeteksi MAKSIMAL 1 orang
        # ==========================================================
        results = model(frame, stream=True, verbose=False, conf=0.6, iou=0.4, max_det=1)

        for r in results:
            annotated_frame = r.plot() # Gambar skeleton bawaan YOLO

            # Validasi keberadaan deteksi manusia
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                
                # Looping ini dijamin hanya berjalan maksimal 1 KALI 
                # karena kita sudah set max_det=1 di atas
                for i in range(len(r.keypoints.xy)):
                    keypoints = r.keypoints.xy[i]
                    
                    # Pastikan 17 titik COCO tersedia dan bounding box valid
                    if len(keypoints) >= 17 and r.boxes is not None and len(r.boxes.xyxy) > i:
                        box = r.boxes.xyxy[i]
                        
                        # Panggil fungsi logika
                        is_fallen, bbox = check_fall_status(keypoints, box)
                        
                        # Anotasi teks
                        annotated_frame = draw_annotations(annotated_frame, is_fallen, bbox)

            # Perekaman
            if SAVE_OUTPUT and out is not None:
                out.write(annotated_frame)

            # Tampilan antarmuka
            cv2.imshow("Sistem Deteksi Jatuh (YOLO Pose)", annotated_frame)

        # Mekanisme keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Program dihentikan pengguna.")
            break

    # Bersihkan memory dan resource
    cap.release()
    if SAVE_OUTPUT and out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")

# Titik mula eksekusi program Python
if __name__ == "__main__":
    main()