import cv2
import socket
import pickle
import struct
import time
import threading
import numpy as np

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
PC_IP        = '192.168.23.140'
PORT         = 9999
RETRY_DELAY  = 3       # seconds between reconnection attempts

# Camera
CAM_INDEX    = 0        # /dev/video0 — change to 1 or 2 if needed
CAM_WIDTH    = 1920
CAM_HEIGHT   = 1080
CAM_FPS      = 15       # 15 fps is plenty for OCR work

# Quality — 95 is the sweet spot: near-lossless for text, ~40% smaller than 98
JPEG_QUALITY = 95

# OCR sharpening strength (0 = disabled, 0.8 = moderate, 1.5 = aggressive)
SHARPEN      = 0.8


# ══════════════════════════════════════════════════════════════
# THREADED CAMERA — always holds the LATEST frame, zero stale lag
# ══════════════════════════════════════════════════════════════
class CameraStream:
    """
    Reads frames in a dedicated thread so the main loop always grabs
    the most recent image instantly — no waiting for USB transfer.
    This alone typically cuts 30-80ms of latency vs a blocking read().
    """

    def __init__(self, index, width, height, fps):
        # Try V4L2 backend first — lower overhead on Linux / Pi OS
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}")

        # Ask camera to output MJPEG — most USB cams have a hardware
        # JPEG encoder, which lets them stream at higher res/fps than
        # raw YUYV and reduces CPU load on the Pi
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,           fps)

        # Keep internal V4L2 buffer as small as possible so we never
        # read a stale queued frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"  Camera opened: {w}×{h} @ {f:.0f} fps")

        self._frame   = None
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # Tight capture loop — runs as fast as the camera provides frames
    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        """Return the latest frame (copy) or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# ══════════════════════════════════════════════════════════════
# OCR PREPROCESSING — lightweight sharpening for text edges
# ══════════════════════════════════════════════════════════════
def sharpen_for_ocr(frame, amount):
    """
    Unsharp-mask sharpening.  Cheap enough for real-time on a Pi 4/5.
    `amount` controls edge emphasis: 0.5 subtle → 1.5 aggressive.
    """
    if amount <= 0:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


# ══════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════
print("═" * 58)
print("  piweb.py  —  OCR-Optimised USB Webcam Sender")
print("═" * 58)

cam = CameraStream(CAM_INDEX, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)

print("  Letting auto-exposure stabilise...")
time.sleep(2)

# Flush any stale frames sitting in the V4L2 buffer
for _ in range(10):
    cam.read()
    time.sleep(0.02)

print("  Camera ready!\n")

# Pre-allocate the JPEG encode param list (avoids per-frame list creation)
_enc = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]


# ══════════════════════════════════════════════════════════════
# MAIN LOOP — reconnects forever, never exits on its own
# ══════════════════════════════════════════════════════════════
try:
    while True:

        # ── 1.  Connect to receiver ────────────────────────────
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)

        try:
            print(f"⏳ Connecting to receiver at {PC_IP}:{PORT} ...")
            sock.connect((PC_IP, PORT))
            sock.settimeout(None)

            # ── TCP tuning for low-latency video ──
            #  TCP_NODELAY: send every packet immediately (disable Nagle)
            #  SO_SNDBUF  : 1 MB send buffer avoids blocking on large frames
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET,  socket.SO_SNDBUF, 1 << 20)

            print("✅ Connected! Streaming...\n")

        except (ConnectionRefusedError, OSError, socket.timeout):
            print(f"❌ Receiver not up. Retrying in {RETRY_DELAY}s...")
            sock.close()
            time.sleep(RETRY_DELAY)
            continue

        # ── 2.  Stream until something breaks ──────────────────
        n_frames = 0
        t0 = time.monotonic()

        try:
            while True:
                frame = cam.read()
                if frame is None:
                    time.sleep(0.005)
                    continue

                # Light sharpening for OCR text edges
                frame = sharpen_for_ocr(frame, SHARPEN)

                # JPEG encode
                ok, buf = cv2.imencode('.jpg', frame, _enc)
                if not ok:
                    continue

                # Pack + send  (same protocol as comms check rx.py expects)
                data = pickle.dumps(buf)
                sock.sendall(struct.pack("Q", len(data)) + data)

                n_frames += 1
                if n_frames % 30 == 0:
                    fps = n_frames / (time.monotonic() - t0)
                    kb  = len(data) / 1024
                    print(f"  📊 {fps:.1f} fps  |  frame #{n_frames}  |  {kb:.0f} KB/frame")

        except Exception as e:
            dt = time.monotonic() - t0
            print(f"\n⚠️  Disconnected after {n_frames} frames ({dt:.1f}s): {e}")
            print(f"   Will reconnect in {RETRY_DELAY}s ...\n")

        finally:
            sock.close()

        time.sleep(RETRY_DELAY)

except KeyboardInterrupt:
    print("\n🛑  Stopped by user.")

finally:
    cam.release()
    print("Cleaned up. Goodbye.")