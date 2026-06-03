"""Buat video sintetis singkat untuk smoke-test pipeline main.py (bukan uji akurasi)."""
import cv2
import numpy as np

W, H = 640, 480
out = cv2.VideoWriter("data/_synthetic_test.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
for i in range(30):
    frame = np.full((H, W, 3), 30, np.uint8)
    x = 50 + i * 15
    cv2.rectangle(frame, (x, 150), (x + 80, 400), (200, 200, 200), -1)
    out.write(frame)
out.release()
print("OK: data/_synthetic_test.mp4 dibuat")
