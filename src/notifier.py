"""
notifier.py — Notifikasi darurat saat jatuh terdeteksi.

Dua kanal:
  1. Alarm suara lokal (winsound di Windows; non-blocking via thread).
  2. Pesan Telegram (teks + snapshot frame) lewat HTTP Bot API.

Semua bersifat OPSIONAL & gagal-dengan-anggun: jika token tak diset atau tak ada
internet, sistem tetap berjalan tanpa error. Token diambil dari environment
variable (lihat config), TIDAK ditulis di kode (aman untuk di-commit).

Cara mengaktifkan Telegram (PowerShell):
    $env:FALLDET_TG_TOKEN = "123456:ABC..."     # token dari @BotFather
    $env:FALLDET_TG_CHAT  = "123456789"         # chat id tujuan
    # lalu set ENABLE_TELEGRAM=True di config.py
"""

import os
import threading
import time

import cv2

import config


class Notifier:
    def __init__(self):
        self.last_notify = {}   # track_id -> timestamp terakhir notifikasi
        self.tg_token = os.environ.get(config.TELEGRAM_BOT_TOKEN_ENV)
        self.tg_chat = os.environ.get(config.TELEGRAM_CHAT_ID_ENV)
        self.telegram_ready = bool(
            config.ENABLE_TELEGRAM and self.tg_token and self.tg_chat
        )
        if config.ENABLE_TELEGRAM and not self.telegram_ready:
            print("[notifier] Telegram diminta tapi token/chat id belum diset "
                  "-> notifikasi Telegram dilewati.")

    # ---- API publik ------------------------------------------------------
    def notify_fall(self, track_id, frame=None):
        """Picu notifikasi untuk satu kejadian jatuh (hormati cooldown per orang)."""
        now = time.time()
        last = self.last_notify.get(track_id, 0.0)
        if now - last < config.NOTIFY_COOLDOWN_SECONDS:
            return
        self.last_notify[track_id] = now

        if config.ENABLE_SOUND_ALARM:
            self._play_alarm_async()
        if self.telegram_ready:
            self._send_telegram_async(track_id, frame)

    # ---- alarm suara -----------------------------------------------------
    def _play_alarm_async(self):
        threading.Thread(target=self._play_alarm, daemon=True).start()

    def _play_alarm(self):
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 250)   # frekuensi Hz, durasi ms
                winsound.Beep(1400, 250)
        except Exception:  # noqa: BLE001 — non-Windows / tanpa audio
            # fallback: bell terminal
            print("\a[ALARM] JATUH TERDETEKSI!")

    # ---- Telegram --------------------------------------------------------
    def _send_telegram_async(self, track_id, frame):
        # salin frame agar thread tidak terganggu modifikasi loop utama
        snapshot = frame.copy() if frame is not None else None
        threading.Thread(
            target=self._send_telegram, args=(track_id, snapshot), daemon=True
        ).start()

    def _send_telegram(self, track_id, frame):
        try:
            import requests
            caption = (
                f"⚠️ DETEKSI JATUH\n"
                f"ID orang: {track_id}\n"
                f"Waktu: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    url = f"https://api.telegram.org/bot{self.tg_token}/sendPhoto"
                    requests.post(
                        url,
                        data={"chat_id": self.tg_chat, "caption": caption},
                        files={"photo": ("fall.jpg", buf.tobytes(), "image/jpeg")},
                        timeout=10,
                    )
                    return
            # tanpa frame -> kirim teks saja
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            requests.post(url, data={"chat_id": self.tg_chat, "text": caption}, timeout=10)
        except Exception as exc:  # noqa: BLE001
            print(f"[notifier] Gagal kirim Telegram: {exc}")
