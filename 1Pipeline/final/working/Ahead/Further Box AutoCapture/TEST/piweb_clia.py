"""
piweb_clia.py  (C525 property-audit — CLI / Keyboard Version)
=====================================================
Raspberry Pi camera streamer with keyboard capture support.
Streams video continuously to the PC pipeline and also lets
you press S on the Pi keyboard to trigger a capture.

This is the "normal" version — no GPIO buttons needed.

Controls (Pi terminal):
    S  →  Send capture signal to PC (triggers pipeline)
    Q  →  Quit

Usage on Pi:
    python piweb_clia.py

Also works on any machine with a webcam for testing.
"""

import cv2
import socket
import pickle
import struct
import time
import json
import threading
import sys
import os
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PC_IP        = '192.168.137.1'
PORT         = 9999
RETRY_DELAY  = 3

# Camera
CAM_INDEX    = 0
CAM_WIDTH    = 1920
CAM_HEIGHT   = 1080
CAM_FPS      = 15
JPEG_QUALITY = 95


# ══════════════════════════════════════════════════════════════════════════════
#  WIRE PROTOCOL — must match pipeline_cli.py
# ══════════════════════════════════════════════════════════════════════════════
MSG_JSON  = 0x01
MSG_FRAME = 0x02

_send_lock = threading.Lock()


def send_json_safe(sock, obj):
    """Thread-safe JSON send."""
    data = json.dumps(obj).encode("utf-8")
    with _send_lock:
        sock.sendall(struct.pack("!BI", MSG_JSON, len(data)) + data)


def send_frame_safe(sock, jpeg_buf):
    """Thread-safe frame send."""
    data = pickle.dumps(jpeg_buf)
    with _send_lock:
        sock.sendall(struct.pack("!BQ", MSG_FRAME, len(data)) + data)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA  (threaded — from original piweb.py)
# ══════════════════════════════════════════════════════════════════════════════
class CameraStream:
    """Reads frames in a dedicated thread — always returns the latest frame."""

    def __init__(self, index, width, height, fps):
        # Try V4L2 backend first (lower overhead on Linux / Pi OS)
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,           fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read-only audit: do not change exposure, focus, sharpness, brightness,
        # or any other C525 control. Camera behavior stays identical to the
        # working piweb_cli.py.
        print("  C525 read-only property audit:")

        def report_actual(prop_name):
            prop_id = getattr(cv2, prop_name, None)
            if prop_id is None:
                print(f"    {prop_name:<24} unavailable in this OpenCV build")
                return None
            actual = self.cap.get(prop_id)
            print(f"    {prop_name:<24} actual={actual:g}")
            return actual

        for prop_name in (
                "CAP_PROP_AUTO_EXPOSURE", "CAP_PROP_EXPOSURE",
                "CAP_PROP_AUTOFOCUS", "CAP_PROP_FOCUS",
                "CAP_PROP_SHARPNESS", "CAP_PROP_BRIGHTNESS"):
            report_actual(prop_name)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
        print(f"  Requested resolution: {width}×{height} @ {fps} fps")
        print(f"  Negotiated resolution: {w}×{h} @ {f:.0f} fps, FOURCC={fourcc!r}")
        if (w, h) != (width, height):
            print("  WARNING: driver did not accept the requested resolution")
        print("  NOTE: cap.get() confirms negotiated output size; use v4l2-ctl "
              "--list-formats-ext to distinguish sensor-native from driver-upscaled modes.")

        self._frame   = None
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def report_settled_controls(self):
        """Print actual controls after auto exposure/focus have seen frames."""
        print("  C525 settled property readback:")
        for prop_name in (
                "CAP_PROP_AUTO_EXPOSURE", "CAP_PROP_EXPOSURE",
                "CAP_PROP_AUTOFOCUS", "CAP_PROP_FOCUS",
                "CAP_PROP_SHARPNESS", "CAP_PROP_BRIGHTNESS"):
            prop_id = getattr(cv2, prop_name, None)
            if prop_id is not None:
                print(f"    {prop_name:<24} actual={self.cap.get(prop_id):g}")

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()




# ══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER (runs in a background thread)
# ══════════════════════════════════════════════════════════════════════════════
_key_queue = None


def _keyboard_listener_windows():
    """Windows: use msvcrt for non-blocking key input."""
    import msvcrt
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
            _key_queue.put(key)
        time.sleep(0.02)


def _keyboard_listener_unix():
    """Unix/Linux: use select + termios for raw non-blocking input."""
    import termios
    import tty
    import select

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0.02)[0]:
                key = sys.stdin.read(1).lower()
                _key_queue.put(key)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def start_keyboard_listener():
    """Start a background thread that reads keyboard input."""
    import queue as _q
    global _key_queue
    _key_queue = _q.Queue()

    if sys.platform == "win32":
        target = _keyboard_listener_windows
    else:
        target = _keyboard_listener_unix

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return _key_queue


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("═" * 58)
    print("  piweb_clia.py — FYDP Smart Glasses (C525 Property Audit)")
    print("═" * 58)
    print("\n  Controls:")
    print("    S  →  Send capture signal to PC")
    print("    Q  →  Quit\n")

    # ── Camera ──
    cam = CameraStream(CAM_INDEX, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
    print("  Letting auto-exposure stabilize …")
    time.sleep(4)
    for _ in range(10):
        cam.read()
        time.sleep(0.02)
    cam.report_settled_controls()
    print("  Camera ready!\n")

    # ── Keyboard listener ──
    key_queue = start_keyboard_listener()
    print("  Keyboard listener active.\n")

    # ── JPEG encode params ──
    _enc = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    # ── Main loop: connect, stream, reconnect ──
    try:
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)

            try:
                print(f"⏳ Connecting to PC at {PC_IP}:{PORT} …")
                sock.connect((PC_IP, PORT))
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
                print("✅ Connected! Streaming …\n")

            except (ConnectionRefusedError, OSError, socket.timeout):
                print(f"❌ PC not ready. Retrying in {RETRY_DELAY}s …")
                sock.close()
                time.sleep(RETRY_DELAY)
                continue

            # ── Stream + keyboard loop ──
            n_frames = 0
            t0 = time.monotonic()

            try:
                while True:
                    # ── Check keyboard ──
                    try:
                        while True:
                            key = key_queue.get_nowait()
                            if key == 'q':
                                print("\n🛑 Q pressed — quitting.")
                                sock.close()
                                cam.release()
                                sys.exit(0)
                            elif key == 's':
                                print("  📸 Capture signal sent to PC!")
                                try:
                                    send_json_safe(sock, {"cmd": "capture_from_pi"})
                                except Exception:
                                    pass
                    except Exception:
                        pass  # queue empty

                    # ── Read & send frame ──
                    frame = cam.read()
                    if frame is None:
                        time.sleep(0.005)
                        continue


                    ok, buf = cv2.imencode('.jpg', frame, _enc)
                    if not ok:
                        continue

                    send_frame_safe(sock, buf)

                    n_frames += 1
                    if n_frames % 30 == 0:
                        fps = n_frames / (time.monotonic() - t0)
                        kb  = len(pickle.dumps(buf)) / 1024
                        print(f"  📊 {fps:.1f} fps  |  frame #{n_frames}  |  {kb:.0f} KB/frame")

            except Exception as e:
                dt = time.monotonic() - t0
                print(f"\n⚠️  Disconnected after {n_frames} frames ({dt:.1f}s): {e}")
                print(f"   Reconnecting in {RETRY_DELAY}s …\n")
            finally:
                sock.close()

            time.sleep(RETRY_DELAY)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cam.release()
        print("Cleaned up. Goodbye.")


if __name__ == "__main__":
    main()
