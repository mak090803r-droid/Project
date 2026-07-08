"""
piweb.py  (Final Demo — GPIO Button Version)
==============================================
Raspberry Pi client with two GPIO buttons, camera, and audio feedback tones.
Connects to the PC pipeline via TCP and responds to JSON commands.

Buttons:
  Button 1 (GPIO 17) — Main control:
    Language:    single=Chinese, double=French, triple=Spanish
    Book mode:   single=on, double=off
    Capture:     single=capture, hold 4s=multi-capture mode
    During TTS:  single tap=pause/resume, double tap=stop

  Button 2 (GPIO 27) — Speed control during TTS:
    Press cycles: 1x → 1.5x → 2x → 1x

Audio Tones (via sounddevice):
  - Short beep on capture
  - Rising tone on OCR success (from PC command)
  - Descending tone on OCR failure (from PC command)

Usage on Pi:
    python piweb.py

If RPi.GPIO is not available (testing on PC), falls back to keyboard input.
"""

import os
import sys
import time
import json
import struct
import pickle
import threading
import socket
import select
import cv2
import numpy as np

# ── GPIO import (graceful fallback for PC testing) ────────────────────────────
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("[WARN] RPi.GPIO not available — using keyboard fallback for testing")

# ── Audio import ──────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("[WARN] sounddevice not available — tones disabled")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PC_IP        = '192.168.23.140'
PORT         = 9999
RETRY_DELAY  = 3

# Buttons (BCM numbering)
BTN1_PIN     = 17    # Main button
BTN2_PIN     = 27    # Speed button

# Camera
CAM_INDEX    = 0
CAM_WIDTH    = 1920
CAM_HEIGHT   = 1080
CAM_FPS      = 15
JPEG_QUALITY = 95
SHARPEN      = 0.8

# Button timing (seconds)
MULTI_PRESS_WINDOW = 0.40   # max gap between presses for multi-tap
HOLD_THRESHOLD     = 4.0    # seconds to trigger "hold"
DEBOUNCE_MS        = 50     # GPIO debounce


# ══════════════════════════════════════════════════════════════════════════════
#  WIRE PROTOCOL — must match PC-side pipeline_nllb1.3b.py
# ══════════════════════════════════════════════════════════════════════════════
MSG_JSON  = 0x01
MSG_FRAME = 0x02

_send_lock = threading.Lock()


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    with _send_lock:
        sock.sendall(struct.pack("!BI", MSG_JSON, len(data)) + data)


def send_frame(sock, jpeg_buf):
    """Send an encoded JPEG frame."""
    data = pickle.dumps(jpeg_buf)
    with _send_lock:
        sock.sendall(struct.pack("!BQ", MSG_FRAME, len(data)) + data)


def recv_msg(sock):
    type_byte = _recv_exact(sock, 1)[0]
    if type_byte == MSG_JSON:
        length = struct.unpack("!I", _recv_exact(sock, 4))[0]
        data   = _recv_exact(sock, length)
        return MSG_JSON, json.loads(data.decode("utf-8"))
    elif type_byte == MSG_FRAME:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
        data   = _recv_exact(sock, length)
        buf    = pickle.loads(data)
        frame  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return MSG_FRAME, frame
    else:
        raise ValueError(f"Unknown message type: 0x{type_byte:02x}")


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA  (from original piweb.py — threaded, always latest frame)
# ══════════════════════════════════════════════════════════════════════════════
class CameraStream:
    """Reads frames in a dedicated thread — always returns the latest frame."""

    def __init__(self, index, width, height, fps):
        # Try V4L2 first, fallback to default
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

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"  Camera opened: {w}×{h} @ {f:.0f} fps")

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

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


def sharpen_for_ocr(frame, amount):
    """Unsharp-mask sharpening for OCR readability."""
    if amount <= 0:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def encode_frame(frame):
    """Capture, sharpen, JPEG-encode. Returns encoded buffer or None."""
    sharpened = sharpen_for_ocr(frame, SHARPEN)
    ok, buf = cv2.imencode('.jpg', sharpened, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf if ok else None


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO TONES
# ══════════════════════════════════════════════════════════════════════════════
def _gen_tone(freq_start, freq_end, duration, sr=22050):
    t    = np.linspace(0, duration, int(sr * duration), False)
    freq = np.linspace(freq_start, freq_end, len(t))
    return (np.sin(2 * np.pi * freq * t) * 0.4).astype(np.float32), sr


def play_beep():
    """Short 1000 Hz beep — confirms capture."""
    if not HAS_AUDIO:
        print("  🔊 *beep*")
        return
    try:
        tone, sr = _gen_tone(1000, 1000, 0.1)
        sd.play(tone, samplerate=sr)
        sd.wait()
    except Exception:
        pass


def play_tone_success():
    """Rising tone 400→800 Hz — OCR found text."""
    if not HAS_AUDIO:
        print("  🔊 *rising tone*")
        return
    try:
        tone, sr = _gen_tone(400, 800, 0.2)
        sd.play(tone, samplerate=sr)
        sd.wait()
    except Exception:
        pass


def play_tone_failure():
    """Descending tone 800→400 Hz — no text detected."""
    if not HAS_AUDIO:
        print("  🔊 *descending tone*")
        return
    try:
        tone, sr = _gen_tone(800, 400, 0.3)
        sd.play(tone, samplerate=sr)
        sd.wait()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  GPIO BUTTON DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def setup_gpio():
    if not HAS_GPIO:
        return
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BTN1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BTN2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print(f"  GPIO ready: BTN1=GPIO{BTN1_PIN}, BTN2=GPIO{BTN2_PIN}")


def _gpio_btn_pressed(pin):
    """Return True if the button on *pin* is currently pressed (active-low)."""
    return GPIO.input(pin) == GPIO.LOW


def _gpio_wait_btn1_pattern():
    """Block until Button 1 produces a pattern.
    Returns: 'single', 'double', 'triple', or 'hold'
    """
    # Wait for first press
    while not _gpio_btn_pressed(BTN1_PIN):
        time.sleep(0.01)
    time.sleep(DEBOUNCE_MS / 1000)  # debounce

    press_start = time.time()

    # Wait for release OR hold threshold
    while _gpio_btn_pressed(BTN1_PIN):
        if time.time() - press_start >= HOLD_THRESHOLD:
            # Still held → "hold"
            while _gpio_btn_pressed(BTN1_PIN):
                time.sleep(0.01)
            return "hold"
        time.sleep(0.01)

    # Released — count additional presses within window
    presses  = 1
    deadline = time.time() + MULTI_PRESS_WINDOW

    while time.time() < deadline:
        if _gpio_btn_pressed(BTN1_PIN):
            time.sleep(DEBOUNCE_MS / 1000)
            if _gpio_btn_pressed(BTN1_PIN):
                presses += 1
                # Wait for release
                while _gpio_btn_pressed(BTN1_PIN):
                    time.sleep(0.01)
                deadline = time.time() + MULTI_PRESS_WINDOW  # reset window
        time.sleep(0.01)

    return {1: "single", 2: "double"}.get(presses, "triple")


def _gpio_detect_tts_btn1():
    """Detect single or double tap of Button 1 (for TTS control).
    Called when a press is already detected. Returns 'single' or 'double'.
    """
    # Wait for release
    while _gpio_btn_pressed(BTN1_PIN):
        time.sleep(0.01)

    # Check for second press within 300ms
    deadline = time.time() + 0.3
    while time.time() < deadline:
        if _gpio_btn_pressed(BTN1_PIN):
            time.sleep(DEBOUNCE_MS / 1000)
            if _gpio_btn_pressed(BTN1_PIN):
                while _gpio_btn_pressed(BTN1_PIN):
                    time.sleep(0.01)
                return "double"
        time.sleep(0.01)
    return "single"


# ── Keyboard fallback for testing on PC ──
def _kb_wait_btn1_pattern():
    print("  [BTN1] Enter: 1=single, 2=double, 3=triple, h=hold: ", end="", flush=True)
    key = input().strip().lower()
    return {"1": "single", "2": "double", "3": "triple", "h": "hold"}.get(key, "single")


def wait_for_btn1_pattern():
    if HAS_GPIO:
        return _gpio_wait_btn1_pattern()
    else:
        return _kb_wait_btn1_pattern()


# ══════════════════════════════════════════════════════════════════════════════
#  TTS CONTROL MODE  (runs during TTS playback)
# ══════════════════════════════════════════════════════════════════════════════
def handle_tts_controls(sock):
    """Monitor both buttons during TTS playback.

    Button 1: single tap → pause/resume, double tap → stop
    Button 2: press → cycle speed (1x → 1.5x → 2x → 1x)

    Exits when the PC sends {"cmd": "tts_ended"}.
    """
    speed_idx = 0
    speeds    = [1.0, 1.5, 2.0]

    while True:
        # ── Check for PC messages (non-blocking) ──
        try:
            readable, _, _ = select.select([sock], [], [], 0.02)
        except (ValueError, OSError):
            return
        if readable:
            try:
                msg_type, msg = recv_msg(sock)
                if msg_type == MSG_JSON:
                    cmd = msg.get("cmd")
                    if cmd == "tts_ended":
                        return
                    elif cmd == "play_tone":
                        tone = msg.get("tone")
                        if tone == "success":
                            play_tone_success()
                        elif tone == "failure":
                            play_tone_failure()
                        elif tone == "capture":
                            play_beep()
            except (ConnectionError, ValueError):
                return

        # ── Check Button 1 ──
        if HAS_GPIO:
            if _gpio_btn_pressed(BTN1_PIN):
                time.sleep(DEBOUNCE_MS / 1000)
                if _gpio_btn_pressed(BTN1_PIN):
                    pattern = _gpio_detect_tts_btn1()
                    if pattern == "single":
                        send_json(sock, {"resp": "tts_control", "action": "pause"})
                    elif pattern == "double":
                        send_json(sock, {"resp": "tts_control", "action": "stop"})
                        # Wait for tts_ended from PC
                        while True:
                            try:
                                msg_type, msg = recv_msg(sock)
                                if msg_type == MSG_JSON and msg.get("cmd") == "tts_ended":
                                    return
                            except (ConnectionError, ValueError):
                                return

            # ── Check Button 2 ──
            if _gpio_btn_pressed(BTN2_PIN):
                time.sleep(DEBOUNCE_MS / 1000)
                if _gpio_btn_pressed(BTN2_PIN):
                    # Wait for release
                    while _gpio_btn_pressed(BTN2_PIN):
                        time.sleep(0.01)
                    speed_idx = (speed_idx + 1) % 3
                    send_json(sock, {"resp": "speed_change", "speed": speeds[speed_idx]})
                    print(f"  🔊 Speed → {speeds[speed_idx]}x")
        else:
            # Keyboard fallback: non-blocking check
            if sys.platform == "win32":
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                    if key == "p":
                        send_json(sock, {"resp": "tts_control", "action": "pause"})
                    elif key == "x":
                        send_json(sock, {"resp": "tts_control", "action": "stop"})
                        while True:
                            try:
                                msg_type, msg = recv_msg(sock)
                                if msg_type == MSG_JSON and msg.get("cmd") == "tts_ended":
                                    return
                            except (ConnectionError, ValueError):
                                return
                    elif key == "s":
                        speed_idx = (speed_idx + 1) % 3
                        send_json(sock, {"resp": "speed_change", "speed": speeds[speed_idx]})
                        print(f"  🔊 Speed → {speeds[speed_idx]}x")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN COMMAND LOOP
# ══════════════════════════════════════════════════════════════════════════════
def connect_to_pc():
    """Connect to the PC pipeline server. Retries forever."""
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            print(f"⏳ Connecting to PC at {PC_IP}:{PORT} …")
            sock.connect((PC_IP, PORT))
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
            print("✅ Connected to PC pipeline!")
            return sock
        except (ConnectionRefusedError, OSError, socket.timeout):
            print(f"❌ PC not ready. Retrying in {RETRY_DELAY}s …")
            sock.close()
            time.sleep(RETRY_DELAY)


def main():
    print("═" * 58)
    print("  piweb.py — FYDP Smart Glasses (Button Mode)")
    print("═" * 58)

    # ── Setup ──
    setup_gpio()

    print("\n[CAM] Initializing camera …")
    cam = CameraStream(CAM_INDEX, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
    print("  Letting auto-exposure stabilize …")
    time.sleep(2)
    for _ in range(10):
        cam.read()
        time.sleep(0.02)
    print("  Camera ready!\n")

    # ── Connect to PC ──
    sock = connect_to_pc()

    # ── Main command loop ──
    try:
        while True:
            msg_type, msg = recv_msg(sock)

            if msg_type != MSG_JSON:
                continue

            cmd = msg.get("cmd")

            # ────────── Language Selection ──────────
            if cmd == "choose_language":
                print("\n🌐 Choose language:")
                print("  Button 1: single=Chinese, double=French, triple=Spanish")
                pattern = wait_for_btn1_pattern()
                lang_map = {
                    "single": "chinese",
                    "double": "french",
                    "triple": "spanish",
                }
                language = lang_map.get(pattern, "chinese")
                print(f"  → Selected: {language}")
                send_json(sock, {"resp": "language", "value": language})

            # ────────── Book Mode Selection ──────────
            elif cmd == "choose_book_mode":
                print("\n📖 Choose book mode:")
                print("  Button 1: single=ON, double=OFF")
                pattern = wait_for_btn1_pattern()
                mode = "on" if pattern == "single" else "off"
                print(f"  → Book mode: {mode}")
                send_json(sock, {"resp": "book_mode", "value": mode})

            # ────────── Capture ──────────
            elif cmd == "wait_capture":
                print("\n📸 Waiting for capture …")
                print("  Button 1: single=capture, hold 4s=multi-capture")
                pattern = wait_for_btn1_pattern()

                if pattern in ("single", "double", "triple"):
                    # Single capture
                    frame = cam.read()
                    if frame is None:
                        print("  [WARN] No frame from camera")
                        continue
                    play_beep()
                    jpeg = encode_frame(frame)
                    if jpeg is None:
                        continue
                    send_json(sock, {"resp": "single_capture"})
                    send_frame(sock, jpeg)
                    print("  ✅ Frame sent to PC")

                elif pattern == "hold":
                    # Multi-capture mode
                    print("  📚 MULTI-CAPTURE MODE")
                    send_json(sock, {"resp": "multi_capture_start"})

                    # Wait for PC acknowledgment
                    msg_type2, msg2 = recv_msg(sock)
                    if msg_type2 == MSG_JSON and msg2.get("cmd") == "multi_capture_ready":
                        frames_sent = 0
                        print("  Press button to capture pages. Hold 4s to finish.")

                        while True:
                            pattern2 = wait_for_btn1_pattern()
                            if pattern2 in ("single", "double", "triple"):
                                frame = cam.read()
                                if frame is not None:
                                    play_beep()
                                    jpeg = encode_frame(frame)
                                    if jpeg is not None:
                                        send_frame(sock, jpeg)
                                        frames_sent += 1
                                        print(f"    📄 Page {frames_sent} captured")
                            elif pattern2 == "hold":
                                break

                        send_json(sock, {"resp": "multi_capture_end", "count": frames_sent})
                        print(f"  ✅ Multi-capture done: {frames_sent} pages sent")

            # ────────── Play Tone ──────────
            elif cmd == "play_tone":
                tone = msg.get("tone")
                if tone == "success":
                    play_tone_success()
                elif tone == "failure":
                    play_tone_failure()
                elif tone == "capture":
                    play_beep()

            # ────────── TTS Control Mode ──────────
            elif cmd == "tts_started":
                print("\n🔊 TTS playing — controls active:")
                if HAS_GPIO:
                    print("  BTN1: tap=pause, double-tap=stop")
                    print("  BTN2: press=cycle speed")
                else:
                    print("  P=pause, X=stop, S=speed")
                handle_tts_controls(sock)
                print("  TTS ended.")

    except ConnectionError:
        print("\n[ERROR] Lost connection to PC")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cam.release()
        sock.close()
        if HAS_GPIO:
            GPIO.cleanup()
        print("Cleaned up. Goodbye.")


if __name__ == "__main__":
    main()
