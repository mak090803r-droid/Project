"""
pipeline_cli_box.py  (Final Demo — CLI / Keyboard + Quality Feedback)
======================================================================
Same as pipeline_cli.py, but with a pre-capture quality feedback loop.

When S is pressed, instead of instantly capturing, a quality-check loop
runs (~0.3 s per iteration) that:
  • Scores the live frame (sharpness / brightness / evenness → 0-100)
  • Detects text bounding boxes and draws them on the preview window
  • Speaks "closer" or "further" if text region is too small / too large
  • Plays a beep when the score crosses ≥ 75 and auto-captures
  • Pressing S again during the loop → force-capture immediately

All existing OCR / Translation / TTS logic is UNCHANGED.

Usage:
    python pipeline_cli_box.py
"""

import os
import sys
import re
import time
import json
import struct
import pickle
import queue
import threading
import socket
import cv2
import numpy as np
import torch
import ctranslate2
import transformers

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))
PICS_DIR     = os.path.join(PROJECT_DIR, "pics")
CAPTURED_DIR = os.path.join(PICS_DIR, "captured")
os.makedirs(CAPTURED_DIR, exist_ok=True)

PORT = 9999

# ══════════════════════════════════════════════════════════════════════════════
#  WIRE PROTOCOL — must match piweb_cli.py
# ══════════════════════════════════════════════════════════════════════════════
MSG_JSON  = 0x01
MSG_FRAME = 0x02

_send_lock = threading.Lock()


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Socket closed by remote")
        buf += chunk
    return buf


def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    with _send_lock:
        sock.sendall(struct.pack("!BI", MSG_JSON, len(data)) + data)


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
#  AUDIO TONES  (PC speakers)
# ══════════════════════════════════════════════════════════════════════════════
def _play_tone(freq_start, freq_end=None, duration=0.15, volume=0.3):
    try:
        import sounddevice as sd
        sr = 22050
        t  = np.linspace(0, duration, int(sr * duration), False)
        if freq_end is None:
            freq_end = freq_start
        freq = np.linspace(freq_start, freq_end, len(t))
        tone = (np.sin(2 * np.pi * freq * t) * volume).astype(np.float32)
        sd.play(tone, samplerate=sr)
        sd.wait()
    except Exception:
        pass

def tone_success():
    _play_tone(400, 800, 0.2)

def tone_failure():
    _play_tone(800, 400, 0.3)

def beep_capture():
    _play_tone(1000, 1000, 0.1)

def beep_ready():
    """Short high-pitched beep signaling frame quality is good enough."""
    _play_tone(1200, 1200, 0.12, volume=0.4)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING  (unchanged from original pipeline)
# ══════════════════════════════════════════════════════════════════════════════
PREPROCESS_CONFIG = {
    "upscale_factor":   1,
    "clahe_clip":       2,
    "clahe_grid":       8,
    "gamma":            0.7,
    "sharpen":          False,
    "sharpen_strength": 1.0,
}

_INV_GAMMA = 1.0 / PREPROCESS_CONFIG["gamma"]
_GAMMA_LUT = np.array([
    ((i / 255.0) ** _INV_GAMMA) * 255 for i in range(256)
]).astype("uint8")

_CLAHE_ENGINE = cv2.createCLAHE(
    clipLimit=PREPROCESS_CONFIG["clahe_clip"],
    tileGridSize=(PREPROCESS_CONFIG["clahe_grid"], PREPROCESS_CONFIG["clahe_grid"]),
)

_S = PREPROCESS_CONFIG["sharpen_strength"]
_SHARPEN_KERNEL = np.array([
    [-_S,    -_S,   -_S],
    [-_S, 1+8*_S,  -_S],
    [-_S,    -_S,   -_S],
])


def preprocess_image(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return None
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray     = cv2.GaussianBlur(gray, (3, 3), 0)
    contrast = _CLAHE_ENGINE.apply(gray)
    enhanced = cv2.LUT(contrast, _GAMMA_LUT)
    if PREPROCESS_CONFIG["sharpen"]:
        enhanced = cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)
    h, w = enhanced.shape[:2]
    return cv2.resize(
        enhanced,
        (int(w * PREPROCESS_CONFIG["upscale_factor"]),
         int(h * PREPROCESS_CONFIG["upscale_factor"])),
        interpolation=cv2.INTER_LINEAR,
    )


def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  NLLB 1.3B  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    NLLB_DEVICE       = "cuda"
    NLLB_COMPUTE_TYPE = "float16"
    os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
    print("[INFO] Hardware Target: NVIDIA GPU (CUDA)")
else:
    NLLB_DEVICE       = "cpu"
    NLLB_COMPUTE_TYPE = "int8"
    print("[INFO] Hardware Target: CPU ONLY (Int8)")

NLLB_MODEL_DIR  = os.path.join(PROJECT_DIR, "nllb-200-1.3B-ct2")
NLLB_BASE_MODEL = "facebook/nllb-200-distilled-1.3B"

NLLB_LANG_MAP = {
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "en": "eng_Latn",
}

PIPELINE_LANG_TO_NLLB = {
    "french":  "fr",
    "chinese": "zh",
    "spanish": "es",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_ocr_engine():
    from paddleocr import PaddleOCR
    ocr_device = "gpu" if NLLB_DEVICE == "cuda" else "cpu"
    print(f"[LOAD] Initializing PaddleOCR (onnxruntime, {ocr_device}) …")
    t = time.time()
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
        device=ocr_device,
    )
    print(f"[LOAD] PaddleOCR ready in {time.time()-t:.2f}s")
    return ocr


def load_unwarper():
    from paddleocr import TextImageUnwarping
    print("[LOAD] Initializing UVDoc unwarper …")
    t = time.time()
    unwarper = TextImageUnwarping(model_name="UVDoc", engine="paddle")
    print(f"[LOAD] UVDoc ready in {time.time()-t:.2f}s")
    return unwarper


def load_nllb_translator(language):
    """Load NLLB-200 1.3B for the given language (skip if English)."""
    if language == "english":
        return None, None
    print("[LOAD] Loading NLLB-200 1.3B Distilled …")
    t = time.time()
    if not os.path.exists(NLLB_MODEL_DIR):
        print(f"[INFO] Compiling {NLLB_BASE_MODEL} → CTranslate2 …")
        cmd = (
            f"ct2-transformers-converter --model {NLLB_BASE_MODEL} "
            f"--output_dir {NLLB_MODEL_DIR} --force --quantization {NLLB_COMPUTE_TYPE}"
        )
        if os.system(cmd) != 0:
            raise RuntimeError("[ERROR] Model compilation failed.")
    translator = ctranslate2.Translator(NLLB_MODEL_DIR, device=NLLB_DEVICE, compute_type=NLLB_COMPUTE_TYPE)
    tokenizer  = transformers.AutoTokenizer.from_pretrained(NLLB_BASE_MODEL)
    print(f"[LOAD] NLLB 1.3B ready in {time.time()-t:.2f}s")
    return translator, tokenizer


def load_tts():
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    import pipertts
    print("[LOAD] Pre-loading Piper TTS voice …")
    t = time.time()
    pipertts.ensure_model()
    pipertts.preload()
    print(f"[LOAD] TTS ready in {time.time()-t:.2f}s")
    return pipertts


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def save_frame(frame, run_count):
    sharpened = unsharp_mask(frame, sigma=1.0, strength=1.5)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename  = f"capture_cli_{run_count:03d}_{timestamp}.jpg"
    filepath  = os.path.join(CAPTURED_DIR, filename)
    # Save as high-quality JPEG (95%) instead of slow PNG
    cv2.imwrite(filepath, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  💾 Saved → {filepath}")
    return filepath


def run_ocr(ocr_engine, img_path, book_mode, unwarper):
    t0 = time.time()
    if book_mode and unwarper is not None:
        print("[STAGE 1] Unwarping curved page …")
        unwarp_result = unwarper.predict(img_path, batch_size=1)
        for res in unwarp_result:
            unwarped_img = res["doctr_img"]
        temp_path = os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")
        cv2.imwrite(temp_path, unwarped_img)
        img_path = temp_path

    print("[STAGE 1] Preprocessing image …")
    processed = preprocess_image(img_path)
    if processed is None:
        return "", time.time() - t0
    temp_proc = os.path.join(PIPELINE_DIR, "_temp_processed.jpg")
    cv2.imwrite(temp_proc, processed)

    print("[STAGE 1] Running PaddleOCR …")
    result = ocr_engine.predict(temp_proc)
    lines = []
    for res in result:
        if "rec_texts" in res:
            for text in res["rec_texts"]:
                if text.strip():
                    lines.append(text.strip())
    extracted = " ".join(lines)

    for f in [temp_proc, os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")]:
        if os.path.exists(f):
            os.remove(f)

    elapsed = time.time() - t0
    print(f"[STAGE 1] OCR complete in {elapsed:.3f}s  ({len(lines)} lines)")
    return extracted, elapsed


def run_translation_nllb(text, language, translator, tokenizer):
    if translator is None or tokenizer is None:
        print("[STAGE 2] Skipping translation (English source)")
        return text, 0.0
    print("[STAGE 2] Translating → English (NLLB 1.3B) …")
    t0 = time.time()
    src_lang_code      = PIPELINE_LANG_TO_NLLB[language]
    src_token          = NLLB_LANG_MAP[src_lang_code]
    tgt_token          = NLLB_LANG_MAP["en"]
    tokenizer.src_lang = src_token

    if src_lang_code == "zh":
        raw_chunks = re.split(r'(。|？|！)', text)
        chunks = []
        for i in range(0, len(raw_chunks)-1, 2):
            chunks.append(raw_chunks[i] + raw_chunks[i+1])
        if len(raw_chunks) % 2 != 0 and raw_chunks[-1].strip():
            chunks.append(raw_chunks[-1])
    else:
        chunks = re.split(r'(?<=[.!?])\s+', text)

    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        return text, time.time() - t0

    tokenized_batch = []
    for chunk in chunks:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk))
        tokenized_batch.append(tokens)
    if not tokenized_batch:
        return text, time.time() - t0

    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    results = translator.translate_batch(tokenized_batch, target_prefix=target_prefixes)

    translated_sentences = []
    for res in results:
        output_tokens  = res.hypotheses[0]
        raw_decoded    = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        clean_sentence = raw_decoded.replace(tgt_token, "").strip()
        translated_sentences.append(clean_sentence)

    result  = " ".join(translated_sentences)
    elapsed = time.time() - t0
    print(f"[STAGE 2] Translation complete in {elapsed:.3f}s")
    return result, elapsed


def run_tts(tts_module, text, key_queue=None, event_queue=None):
    """Speak text. Press any key on PC or Pi to stop early."""
    print("[STAGE 3] Speaking … (press any key to stop)")
    t0 = time.time()

    # Clear out any old keystrokes/commands before starting playback
    if key_queue is not None:
        try:
            while not key_queue.empty():
                key_queue.get_nowait()
        except queue.Empty:
            pass

    if event_queue is not None:
        try:
            while not event_queue.empty():
                event_queue.get_nowait()
        except queue.Empty:
            pass

    tts_module.speak(text)

    stopped = False
    while tts_module.is_speaking():
        # Check PC keyboard queue
        if key_queue is not None and not key_queue.empty():
            try:
                while not key_queue.empty():
                    key_queue.get_nowait()
                tts_module.stop()
                stopped = True
                break
            except queue.Empty:
                pass

        # Check Pi command event queue
        if event_queue is not None and not event_queue.empty():
            try:
                stop_signal = False
                while not event_queue.empty():
                    evt = event_queue.get_nowait()
                    if evt.get("cmd") == "capture_from_pi" or evt.get("event") == "disconnect":
                        stop_signal = True
                if stop_signal:
                    tts_module.stop()
                    stopped = True
                    break
            except queue.Empty:
                pass

        time.sleep(0.05)

    tts_module.wait_until_done()

    # Clear any keystrokes/commands typed during playback to prevent queue build-up looping
    if key_queue is not None:
        try:
            while not key_queue.empty():
                key_queue.get_nowait()
        except queue.Empty:
            pass

    if event_queue is not None:
        try:
            while not event_queue.empty():
                event_queue.get_nowait()
        except queue.Empty:
            pass

    elapsed = time.time() - t0
    if stopped:
        print(f"[STAGE 3] TTS stopped after {elapsed:.3f}s")
    else:
        print(f"[STAGE 3] TTS finished in {elapsed:.3f}s")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  RECEIVER THREAD  — keeps the latest frame + capture signals in queues
# ══════════════════════════════════════════════════════════════════════════════
class FrameHolder:
    """Thread-safe holder for the most recent camera frame from the Pi."""
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()

    def update(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


def receiver_loop(sock, frame_holder, event_queue):
    """Background thread: reads messages from Pi, updates frame_holder and event_queue."""
    while True:
        try:
            msg_type, data = recv_msg(sock)
            if msg_type == MSG_FRAME:
                frame_holder.update(data)
            elif msg_type == MSG_JSON:
                event_queue.put(data)
        except (ConnectionError, OSError, ValueError):
            event_queue.put({"event": "disconnect"})
            break


# ══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER  (PC side — S to capture, Q to quit)
# ══════════════════════════════════════════════════════════════════════════════
def start_keyboard_listener(key_queue):
    """Start a background thread that reads keyboard input on PC."""
    def _listener_windows():
        import msvcrt
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                key_queue.put(key)
            time.sleep(0.02)

    def _listener_unix():
        import termios, tty, select
        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    key = sys.stdin.read(1).lower()
                    key_queue.put(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    t = threading.Thread(
        target=_listener_windows if sys.platform == "win32" else _listener_unix,
        daemon=True)
    t.start()


# ══════════════════════════════════════════════════════════════════════════════
#  TCP SERVER
# ══════════════════════════════════════════════════════════════════════════════
def start_server(port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    print(f"\n[SERVER] Listening on port {port} …")
    print("  Waiting for Pi to connect …\n")
    client, addr = server.accept()
    print(f"[SERVER] ✅ Pi connected from {addr}")
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    return server, client


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME QUALITY SCORING  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
QUALITY_THRESHOLD = 75   # score ≥ this → auto-capture

def score_frame_quality(frame):
    """
    Score a camera frame from 0-100 based on:
      - Sharpness  (40 pts) — Laplacian variance
      - Brightness (30 pts) — mean pixel value, ideal 80-180
      - Evenness   (30 pts) — brightness consistency across 4 quadrants

    Returns: (score: int, details: dict)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Sharpness (40 pts) — Laplacian variance ──
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        sharpness_pts = 0
    elif laplacian_var >= 200:
        sharpness_pts = 40
    else:
        sharpness_pts = int((laplacian_var - 50) / 150 * 40)

    # ── Brightness (30 pts) — mean pixel value, ideal range 80–180 ──
    mean_brightness = float(np.mean(gray))
    if 80 <= mean_brightness <= 180:
        brightness_pts = 30
    elif mean_brightness < 80:
        brightness_pts = max(0, int(mean_brightness / 80 * 30))
    else:
        brightness_pts = max(0, int((255 - mean_brightness) / 75 * 30))

    # ── Evenness (30 pts) — quadrant brightness std-dev ──
    mid_h, mid_w = h // 2, w // 2
    quadrants = [
        gray[:mid_h, :mid_w],
        gray[:mid_h, mid_w:],
        gray[mid_h:, :mid_w],
        gray[mid_h:, mid_w:],
    ]
    quad_means = [float(np.mean(q)) for q in quadrants]
    quad_std   = float(np.std(quad_means))
    # Low std = even lighting.  std < 5 → full marks, std > 40 → 0
    if quad_std < 5:
        evenness_pts = 30
    elif quad_std > 40:
        evenness_pts = 0
    else:
        evenness_pts = int((40 - quad_std) / 35 * 30)

    score = sharpness_pts + brightness_pts + evenness_pts
    details = {
        "sharpness": round(laplacian_var, 1),
        "sharpness_pts": sharpness_pts,
        "brightness": round(mean_brightness, 1),
        "brightness_pts": brightness_pts,
        "evenness_std": round(quad_std, 1),
        "evenness_pts": evenness_pts,
    }
    return min(100, score), details


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT REGION DETECTION FOR CLOSER/FURTHER HINTS  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def detect_text_region_size(frame, ocr_engine):
    """
    Run lightweight text detection on a downscaled frame to decide
    whether to say 'closer' or 'further'.

    Returns: ("closer" | "further" | None, boxes_for_display)
      - boxes_for_display: list of 4-point polygons scaled back to
        original frame size, for drawing on the preview.
    """
    h_orig, w_orig = frame.shape[:2]

    # Downscale to ~320px wide for speed
    scale = 320.0 / w_orig
    small = cv2.resize(frame, (320, int(h_orig * scale)))
    h_s, w_s = small.shape[:2]

    # Write to temp file for PaddleOCR
    temp_det = os.path.join(PIPELINE_DIR, "_temp_det_preview.jpg")
    cv2.imwrite(temp_det, small, [cv2.IMWRITE_JPEG_QUALITY, 80])

    try:
        result = ocr_engine.predict(temp_det)
    except Exception:
        return None, []
    finally:
        if os.path.exists(temp_det):
            os.remove(temp_det)

    # Extract bounding box polygons
    boxes_small = []
    for res in result:
        if "dt_polys" in res:
            for poly in res["dt_polys"]:
                boxes_small.append(np.array(poly, dtype=np.float32))

    if not boxes_small:
        return "closer", []   # No text at all → move closer

    # Scale boxes back to original frame size
    inv_scale = 1.0 / scale
    boxes_orig = []
    for poly in boxes_small:
        scaled_poly = (poly * inv_scale).astype(np.int32)
        boxes_orig.append(scaled_poly)

    # Compute total bounding box area as fraction of frame area
    frame_area = h_orig * w_orig
    total_box_area = 0
    touching_edges = 0
    edge_margin = int(10 * inv_scale)

    for poly in boxes_orig:
        # Bounding rect area
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        total_box_area += (x_max - x_min) * (y_max - y_min)

        # Check if box touches frame edges
        if x_min < edge_margin or y_min < edge_margin or \
           x_max > w_orig - edge_margin or y_max > h_orig - edge_margin:
            touching_edges += 1

    area_ratio = total_box_area / frame_area

    hint = None
    if area_ratio < 0.03:
        hint = "closer"
    elif area_ratio > 0.85 and touching_edges >= 2:
        hint = "further"

    return hint, boxes_orig


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY OVERLAY ON PREVIEW  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def draw_quality_overlay(frame, score, details, boxes):
    """
    Draw quality info + bounding boxes on a COPY of the frame for
    the preview window. Does NOT modify the original frame.

    Returns the annotated copy.
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # ── Draw bounding boxes ──
    for poly in boxes:
        pts = poly.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # ── Score bar at top ──
    bar_h = 40
    # Semi-transparent dark background
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Score colour
    if score >= QUALITY_THRESHOLD:
        color = (0, 220, 0)     # green
    elif score >= 50:
        color = (0, 200, 255)   # yellow/orange
    else:
        color = (0, 0, 220)     # red

    # Score bar fill
    bar_w = int((score / 100) * (w - 20))
    cv2.rectangle(display, (10, 8), (10 + bar_w, bar_h - 8), color, -1)

    # Score text
    cv2.putText(display, f"Quality: {score}/100", (15, bar_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Details text on the right
    detail_str = (f"Sharp:{details['sharpness_pts']}  "
                  f"Bright:{details['brightness_pts']}  "
                  f"Even:{details['evenness_pts']}")
    text_size = cv2.getTextSize(detail_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(display, detail_str, (w - text_size[0] - 15, bar_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return display


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-CAPTURE QUALITY LOOP  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def pre_capture_quality_loop(frame_holder, ocr_engine, tts_module,
                              key_queue, event_queue):
    """
    Quality-check loop that runs after the user presses S but BEFORE
    the frame is sent to the OCR/Translation/TTS pipeline.

    Checks frame quality every ~0.3s.
    Runs text detection and speaks "closer"/"further" max once per 1.5s.
    Auto-captures when score >= QUALITY_THRESHOLD (plays beep_ready).
    Pressing S again → force-captures immediately.
    Pressing Q → aborts, returns None.

    Returns: frame (np.ndarray) to capture, or None to abort.
    """
    print("\n[QUALITY] Entering pre-capture quality check …")
    print(f"[QUALITY] Target score: ≥ {QUALITY_THRESHOLD}/100")
    print("[QUALITY] Press S to force-capture, Q to cancel\n")

    last_voice_time  = 0.0      # throttle voice feedback
    last_detect_time = 0.0      # throttle text detection
    voice_cooldown   = 1.5      # seconds between voice hints
    detect_interval  = 1.5      # seconds between text detection runs
    check_interval   = 0.3      # seconds between quality score checks

    cached_boxes = []           # last detected boxes for overlay
    cached_hint  = None         # last hint result

    while True:
        frame = frame_holder.get()
        if frame is None:
            time.sleep(check_interval)
            continue

        # ── Score the frame ──
        score, details = score_frame_quality(frame)

        # ── Run text detection (throttled) ──
        now = time.time()
        if now - last_detect_time >= detect_interval:
            last_detect_time = now
            cached_hint, cached_boxes = detect_text_region_size(frame, ocr_engine)

        # ── Draw overlay on preview ──
        display = draw_quality_overlay(frame, score, details, cached_boxes)
        cv2.imshow("Smart Glasses Live Stream", cv2.resize(display, (640, 360)))
        cv2.waitKey(1)

        # ── Voice feedback (throttled) ──
        if cached_hint is not None and (now - last_voice_time >= voice_cooldown):
            last_voice_time = time.time()
            if cached_hint == "closer":
                tts_module.speak("move closer.")
                tts_module.wait_until_done()
            elif cached_hint == "further":
                tts_module.speak("back up.")
                tts_module.wait_until_done()

        # ── Quality threshold met → auto-capture ──
        if score >= QUALITY_THRESHOLD:
            print(f"[QUALITY] ✅ Score {score} ≥ {QUALITY_THRESHOLD} — auto-capturing!")
            beep_ready()
            return frame

        # ── Check for Q (abort) or S (force-capture) from PC keyboard ──
        try:
            while not key_queue.empty():
                key = key_queue.get_nowait()
                if key == 'q':
                    print("[QUALITY] ❌ Cancelled by user (Q)")
                    return None
                elif key == 's':
                    print(f"[QUALITY] ⚡ Force-capture (S) at score {score}")
                    beep_ready()
                    return frame
        except queue.Empty:
            pass

        # ── Check for Pi events ──
        try:
            while not event_queue.empty():
                evt = event_queue.get_nowait()
                if evt.get("event") == "disconnect":
                    print("[QUALITY] Pi disconnected — aborting")
                    return None
                if evt.get("cmd") == "capture_from_pi":
                    print(f"[QUALITY] ⚡ Force-capture from Pi (S) at score {score}")
                    beep_ready()
                    return frame
        except queue.Empty:
            pass

        time.sleep(check_interval)


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS & SPEAK  (single frame — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def process_and_speak(frame, run_count, language, book_mode,
                      ocr_engine, unwarper, translator, tokenizer, tts_module,
                      key_queue=None, event_queue=None):
    print(f"\n{'─'*60}")
    print(f"  RUN #{run_count}")
    print(f"{'─'*60}\n")

    pipeline_t0 = time.time()
    beep_capture()
    # Save frame to disk
    img_path = save_frame(frame, run_count)

    # Show the captured frame in a separate popup window
    cv2.imshow("Captured Image (OCR Target)", cv2.resize(frame, (640, 360)))
    cv2.waitKey(1)

    # Stage 1: OCR
    extracted_text, t_ocr = run_ocr(ocr_engine, img_path, book_mode, unwarper)
    if not extracted_text:
        tone_failure()
        tts_module.speak("No text detected. Please try again.")
        tts_module.wait_until_done()
        print("[WARN] No text extracted. Skipping.\n")
        return

    tone_success()

    print(f"\n{'='*60}")
    print("  EXTRACTED TEXT")
    print(f"{'='*60}")
    print(extracted_text)
    print(f"{'='*60}\n")

    # Stage 2: Translation
    english_text, t_translate = run_translation_nllb(
        extracted_text, language, translator, tokenizer)

    if language != "english":
        print(f"\n{'='*60}")
        print("  TRANSLATED TEXT (English) — NLLB 1.3B")
        print(f"{'='*60}")
        print(english_text)
        print(f"{'='*60}\n")

    # Stage 3: TTS
    t_tts = run_tts(tts_module, english_text, key_queue, event_queue)

    total_pipeline = time.time() - pipeline_t0

    print(f"\n{'─'*40}")
    print("  ⏱  TIMING BREAKDOWN")
    print(f"{'─'*40}")
    print(f"  OCR          : {t_ocr:.3f}s")
    if language != "english":
        print(f"  Translation  : {t_translate:.3f}s")
    else:
        print(f"  Translation  : skipped")
    print(f"  TTS Playback : {t_tts:.3f}s")
    print(f"{'─'*40}")
    print(f"  TOTAL        : {total_pipeline:.3f}s")
    print(f"{'─'*40}")
    print(f"\n  Ready for next capture!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║    FYDP SMART GLASSES — FINAL DEMO PIPELINE (CLI + BOX)    ║
║          OCR → TRANSLATE (NLLB 1.3B) → SPEAK               ║
║                                                              ║
║  Controls:                                                   ║
║      S  =  enter quality check → auto-capture               ║
║      Q  =  quit                                              ║
║  Also works with S/Q on the Pi keyboard!                     ║
║                                                              ║
║  Quality Feedback:                                           ║
║      • Bounding boxes drawn on preview                       ║
║      • Score bar at top of preview (green ≥ 75)              ║
║      • Voice: "closer" / "back up" if text too small/large   ║
║      • Beep when quality OK → auto-captures                  ║
╚══════════════════════════════════════════════════════════════╝
"""


class SuppressOutput:
    """Context manager to suppress console spam from heavy libraries during load."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def ask_language():
    print("\n── Select source language ──")
    print("  1) French")
    print("  2) Chinese")
    print("  3) Spanish")
    print("  4) English (no translation, OCR + TTS only)")
    while True:
        choice = input("\nEnter 1/2/3/4: ").strip()
        if choice == "1": return "french"
        if choice == "2": return "chinese"
        if choice == "3": return "spanish"
        if choice == "4": return "english"
        print("[!] Invalid choice.")


def ask_book_mode():
    print("\n── Book Mode (curved page unwarping) ──")
    while True:
        choice = input("Enable book mode? (y/n): ").strip().lower()
        if choice in ("y", "yes"): return True
        if choice in ("n", "no"):  return False
        print("[!] Enter y or n.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(BANNER)

    # ── 1. Welcome & Audio Setup ──
    print("[INFO] Initializing audio subsystem …")
    with SuppressOutput():
        tts_module = load_tts()

    tts_module.speak("Welcome to F Y D P Glasses")

    # ── 2. Load OCR + UVDoc while welcome plays ──
    print("[INFO] Loading OCR and unwarping models (silently) …")
    with SuppressOutput():
        ocr_engine = load_ocr_engine()
        unwarper   = load_unwarper()

    tts_module.wait_until_done()

    # ── 3. CLI Configuration with Speech Guidance ──
    tts_module.speak("Choose language")
    language = ask_language()

    tts_module.speak("Choose book mode")
    book_mode = ask_book_mode()

    if not book_mode:
        unwarper = None

    # ── 4. Load NLLB translator ──
    print(f"\n[INFO] Loading NLLB Translator for {language.upper()} …")
    with SuppressOutput():
        translator, tokenizer = load_nllb_translator(language)

    tts_module.speak("All models loaded")
    tts_module.wait_until_done()

    # ── 5. Connect to Pi ──
    server_sock, client_sock = start_server(PORT)
    tts_module.speak("Smart glasses connected. Press S to start translating.")
    tts_module.wait_until_done()

    # ── 6. Start receiver thread ──
    frame_holder = FrameHolder()
    event_queue  = queue.Queue()
    recv_thread  = threading.Thread(
        target=receiver_loop,
        args=(client_sock, frame_holder, event_queue),
        daemon=True)
    recv_thread.start()

    # ── 7. Start PC keyboard listener ──
    key_queue = queue.Queue()
    start_keyboard_listener(key_queue)

    pipeline_busy = threading.Event()   # set when pipeline is running

    print("\n⏳ Waiting for capture … (S = capture, Q = quit)\n")

    # ── 8. Main loop ──
    run_count = 0

    try:
        while True:
            # ── Display Live Video Stream ──
            if not pipeline_busy.is_set():
                live_frame = frame_holder.get()
                if live_frame is not None:
                    cv2.imshow("Smart Glasses Live Stream", cv2.resize(live_frame, (640, 360)))
                    cv2.waitKey(1)

            # ── Check for disconnect ──
            try:
                event = event_queue.get_nowait()
                if event.get("event") == "disconnect" or event.get("cmd") == "quit":
                    print("\n[INFO] Pi disconnected.")
                    break
                # Pi pressed S → enter quality loop then pipeline
                if event.get("cmd") == "capture_from_pi":
                    if not pipeline_busy.is_set():
                        pipeline_busy.set()
                        frame = pre_capture_quality_loop(
                            frame_holder, ocr_engine, tts_module,
                            key_queue, event_queue)
                        if frame is not None:
                            run_count += 1
                            process_and_speak(
                                frame, run_count, language, book_mode,
                                ocr_engine, unwarper, translator, tokenizer, tts_module,
                                key_queue, event_queue)
                        pipeline_busy.clear()
            except queue.Empty:
                pass

            # ── Check PC keyboard ──
            try:
                key = key_queue.get_nowait()
                if key == 'q':
                    print("\n🛑 Q pressed — quitting.")
                    break
                elif key == 's':
                    if not pipeline_busy.is_set():
                        print("  📸 Capture triggered from PC keyboard!")
                        pipeline_busy.set()
                        frame = pre_capture_quality_loop(
                            frame_holder, ocr_engine, tts_module,
                            key_queue, event_queue)
                        if frame is not None:
                            run_count += 1
                            process_and_speak(
                                frame, run_count, language, book_mode,
                                ocr_engine, unwarper, translator, tokenizer, tts_module,
                                key_queue, event_queue)
                        pipeline_busy.clear()
            except queue.Empty:
                pass

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user.")
    finally:
        tts_module.speak("Goodbye")
        tts_module.wait_until_done()
        try: client_sock.close()
        except Exception: pass
        try: server_sock.close()
        except Exception: pass
        cv2.destroyAllWindows()
        print(f"\n[EXIT] Session ended. Total runs: {run_count}")


if __name__ == "__main__":
    main()
