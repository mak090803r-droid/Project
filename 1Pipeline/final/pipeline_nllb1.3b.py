"""
pipeline_nllb1.3b.py  (Final Demo — GPIO Button Version)
==========================================================
PC-side TCP server that drives the full demo flow via JSON commands
to the Raspberry Pi. The Pi has two GPIO buttons; this file sends
commands and receives button responses + captured frames.

Demo Flow:
  1. Load TTS → "Welcome to FYDP Glasses"
  2. Load OCR, NLLB 1.3B, UVDoc
  3. "Models loaded"
  4. Wait for Pi to connect
  5. "Choose language"  → Button 1 on Pi (single/double/triple)
  6. "Choose book mode" → Button 1 on Pi (single/double)
  7. "Press button to start translating"
  8. Capture loop with multi-capture, TTS controls, speed cycling

Usage:
    python pipeline_nllb1.3b.py
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
os.makedirs(PICS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PORT = 9999

# ══════════════════════════════════════════════════════════════════════════════
#  WIRE PROTOCOL — shared with piweb.py
# ══════════════════════════════════════════════════════════════════════════════
MSG_JSON  = 0x01
MSG_FRAME = 0x02

_send_lock = threading.Lock()


def _recv_exact(sock, n):
    """Receive exactly *n* bytes from the socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Socket closed by remote")
        buf += chunk
    return buf


def send_json(sock, obj):
    """Send a length-prefixed JSON message."""
    data = json.dumps(obj).encode("utf-8")
    with _send_lock:
        sock.sendall(struct.pack("!BI", MSG_JSON, len(data)) + data)


def recv_msg(sock):
    """Receive one message. Returns (msg_type, payload).
    JSON  → payload is a dict.
    FRAME → payload is a BGR numpy array (or None if decode failed).
    """
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
#  AUDIO TONES (PC speakers — optional confirmation for the operator)
# ══════════════════════════════════════════════════════════════════════════════
def _play_tone(freq_start, freq_end=None, duration=0.15, volume=0.3):
    """Generate and play a short tone on the default audio device."""
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
        pass   # non-critical


def tone_success():
    _play_tone(400, 800, 0.2)

def tone_failure():
    _play_tone(800, 400, 0.3)


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
    """Grayscale preprocessing: denoise → CLAHE → gamma → sharpen → upscale."""
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
    final = cv2.resize(
        enhanced,
        (int(w * PREPROCESS_CONFIG["upscale_factor"]),
         int(h * PREPROCESS_CONFIG["upscale_factor"])),
        interpolation=cv2.INTER_LINEAR,
    )
    return final


def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred   = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# ══════════════════════════════════════════════════════════════════════════════
#  NLLB 1.3B  CONFIGURATION  (+ Spanish for triple-press)
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

# FLORES-200 codes
NLLB_LANG_MAP = {
    "zh": "zho_Hans",   # Chinese Simplified
    "fr": "fra_Latn",   # French
    "es": "spa_Latn",   # Spanish
    "en": "eng_Latn",   # English
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


def load_nllb_translator():
    """Load NLLB-200 1.3B — always loaded (language configured later)."""
    print("[LOAD] Loading NLLB-200 1.3B Distilled …")
    t = time.time()

    if not os.path.exists(NLLB_MODEL_DIR):
        print(f"[INFO] Compiling {NLLB_BASE_MODEL} → CTranslate2 …")
        start_conv = time.time()
        cmd = (
            f"ct2-transformers-converter --model {NLLB_BASE_MODEL} "
            f"--output_dir {NLLB_MODEL_DIR} --force --quantization {NLLB_COMPUTE_TYPE}"
        )
        if os.system(cmd) != 0:
            raise RuntimeError("[ERROR] Model compilation failed.")
        print(f"[INFO] Saved to '{NLLB_MODEL_DIR}' ({time.time()-start_conv:.2f}s)")

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
#  PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════
def save_frame(frame, run_count):
    """Save captured frame to PICS_DIR. Returns file path."""
    sharpened = unsharp_mask(frame, sigma=1.0, strength=1.5)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename  = f"capture_{run_count:03d}_{timestamp}.png"
    filepath  = os.path.join(PICS_DIR, filename)
    cv2.imwrite(filepath, sharpened)
    print(f"  💾 Saved capture → {filepath}")
    return filepath


def run_ocr(ocr_engine, img_path, book_mode, unwarper):
    """Run OCR on an image. Returns (extracted_text, elapsed_seconds)."""
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

    # Cleanup temp files
    for f in [temp_proc, os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")]:
        if os.path.exists(f):
            os.remove(f)

    elapsed = time.time() - t0
    print(f"[STAGE 1] OCR complete in {elapsed:.3f}s  ({len(lines)} lines)")
    return extracted, elapsed


def run_translation_nllb(text, language, translator, tokenizer):
    """Translate text → English using NLLB 1.3B. Returns (translated, elapsed)."""
    if translator is None or tokenizer is None:
        print("[STAGE 2] Skipping translation (English source)")
        return text, 0.0

    print("[STAGE 2] Translating → English (NLLB 1.3B) …")
    t0 = time.time()

    src_lang_code      = PIPELINE_LANG_TO_NLLB[language]
    src_token          = NLLB_LANG_MAP[src_lang_code]
    tgt_token          = NLLB_LANG_MAP["en"]
    tokenizer.src_lang = src_token

    # Sentence segmentation
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

    # Batch tokenize
    tokenized_batch = []
    for chunk in chunks:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk))
        tokenized_batch.append(tokens)

    if not tokenized_batch:
        return text, time.time() - t0

    # Translate
    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    results = translator.translate_batch(tokenized_batch, target_prefix=target_prefixes)

    # Decode
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


# ══════════════════════════════════════════════════════════════════════════════
#  TTS WITH PI BUTTON CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
def wait_for_tts_with_controls(tts_module, recv_queue, client_sock):
    """Monitor TTS playback and handle control messages from Pi.

    During TTS:
      - Button 1 single tap  → toggle pause/resume
      - Button 1 double tap  → stop TTS (return to capture)
      - Button 2 press       → cycle speed 1x→1.5x→2x→1x

    Returns True if TTS was stopped by user, False if finished naturally.
    """
    paused = False

    while tts_module.is_speaking():
        try:
            msg_type, data = recv_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if msg_type != MSG_JSON:
            continue

        resp = data.get("resp")

        if resp == "tts_control":
            action = data.get("action")
            if action == "pause":
                if paused:
                    tts_module.resume()
                    paused = False
                else:
                    tts_module.pause()
                    paused = True
            elif action == "stop":
                tts_module.stop()
                tts_module.wait_until_done()
                return True

        elif resp == "speed_change":
            tts_module.set_speed(data.get("speed", 1.0))

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  RECEIVER THREAD + QUEUE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def receiver_loop(sock, recv_queue):
    """Background thread: reads all messages from Pi, puts in queue."""
    while True:
        try:
            msg_type, data = recv_msg(sock)
            recv_queue.put((msg_type, data))
        except (ConnectionError, OSError, ValueError):
            recv_queue.put((None, None))   # signal disconnect
            break


def wait_for_json(recv_queue, expected_resp=None, timeout=None):
    """Block until a JSON message arrives. If expected_resp is set, filter for it."""
    while True:
        msg_type, data = recv_queue.get(timeout=timeout)
        if msg_type is None:
            raise ConnectionError("Pi disconnected")
        if msg_type == MSG_JSON:
            if expected_resp is None or data.get("resp") == expected_resp:
                return data
        # Ignore non-matching messages


def wait_for_frame(recv_queue, timeout=60):
    """Block until a FRAME message arrives."""
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for frame")
        msg_type, data = recv_queue.get(timeout=remaining)
        if msg_type is None:
            raise ConnectionError("Pi disconnected")
        if msg_type == MSG_FRAME:
            return data


def collect_multi_frames(recv_queue, timeout=120):
    """Collect frames during multi-capture until multi_capture_end."""
    frames = []
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            print(f"[WARN] Multi-capture timeout — got {len(frames)} frames")
            break
        msg_type, data = recv_queue.get(timeout=remaining)
        if msg_type is None:
            raise ConnectionError("Pi disconnected")
        if msg_type == MSG_FRAME:
            frames.append(data)
            print(f"  📷 Multi-capture page {len(frames)} received")
        elif msg_type == MSG_JSON and data.get("resp") == "multi_capture_end":
            break
    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  TCP SERVER
# ══════════════════════════════════════════════════════════════════════════════
def start_server(port):
    """Start TCP server, wait for Pi to connect. Returns (server_sock, client_sock)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)

    print(f"\n[SERVER] Listening on port {port} …")
    print("  Waiting for Smart Glasses (Raspberry Pi) to connect …\n")

    client, addr = server.accept()
    print(f"[SERVER] ✅ Pi connected from {addr}")

    # TCP tuning
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

    return server, client


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS & SPEAK  (single frame)
# ══════════════════════════════════════════════════════════════════════════════
def process_and_speak(frame, run_count, language, book_mode,
                      ocr_engine, unwarper, translator, tokenizer,
                      tts_module, recv_queue, client_sock):
    """Full pipeline on a single frame: save → OCR → translate → TTS."""
    print(f"\n{'─'*60}")
    print(f"  RUN #{run_count}")
    print(f"{'─'*60}\n")

    pipeline_t0 = time.time()

    # Save frame to disk
    img_path = save_frame(frame, run_count)

    # Stage 1: OCR
    extracted_text, t_ocr = run_ocr(ocr_engine, img_path, book_mode, unwarper)

    if not extracted_text:
        send_json(client_sock, {"cmd": "play_tone", "tone": "failure"})
        tone_failure()
        tts_module.speak("No text detected. Please try again.")
        tts_module.wait_until_done()
        return

    send_json(client_sock, {"cmd": "play_tone", "tone": "success"})
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

    # Stage 3: TTS with Pi button controls
    print("[STAGE 3] Speaking … (Pi buttons: tap=pause, double-tap=stop)")
    t_tts_start = time.time()

    send_json(client_sock, {"cmd": "tts_started"})
    tts_module.speak(english_text)

    stopped = wait_for_tts_with_controls(tts_module, recv_queue, client_sock)

    tts_module.wait_until_done()
    send_json(client_sock, {"cmd": "tts_ended"})

    t_tts = time.time() - t_tts_start
    total_pipeline = time.time() - pipeline_t0

    # Timing summary
    print(f"\n{'─'*40}")
    print("  ⏱  TIMING BREAKDOWN")
    print(f"{'─'*40}")
    print(f"  OCR          : {t_ocr:.3f}s")
    if language != "english":
        print(f"  Translation  : {t_translate:.3f}s")
    else:
        print(f"  Translation  : skipped")
    print(f"  TTS Playback : {t_tts:.3f}s {'(stopped)' if stopped else ''}")
    print(f"{'─'*40}")
    print(f"  TOTAL        : {total_pipeline:.3f}s")
    print(f"{'─'*40}")


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS & SPEAK  (multi-page)
# ══════════════════════════════════════════════════════════════════════════════
def process_multi_and_speak(frames, run_count, language, book_mode,
                            ocr_engine, unwarper, translator, tokenizer,
                            tts_module, recv_queue, client_sock):
    """Process multiple captured pages: OCR all → concatenate → translate → TTS."""
    print(f"\n{'─'*60}")
    print(f"  RUN #{run_count}  |  Multi-page ({len(frames)} pages)")
    print(f"{'─'*60}\n")

    pipeline_t0 = time.time()
    all_text = []
    t_ocr_total = 0

    for i, frame in enumerate(frames):
        print(f"\n  ── Page {i+1}/{len(frames)} ──")
        img_path = save_frame(frame, f"{run_count}_p{i+1}")
        text, t_ocr = run_ocr(ocr_engine, img_path, book_mode, unwarper)
        t_ocr_total += t_ocr
        if text:
            all_text.append(text)

    combined_text = " ".join(all_text)

    if not combined_text:
        send_json(client_sock, {"cmd": "play_tone", "tone": "failure"})
        tone_failure()
        tts_module.speak("No text detected on any page. Please try again.")
        tts_module.wait_until_done()
        return

    send_json(client_sock, {"cmd": "play_tone", "tone": "success"})
    tone_success()

    print(f"\n{'='*60}")
    print(f"  COMBINED TEXT ({len(all_text)} pages)")
    print(f"{'='*60}")
    print(combined_text[:500] + ("…" if len(combined_text) > 500 else ""))
    print(f"{'='*60}\n")

    # Translation
    english_text, t_translate = run_translation_nllb(
        combined_text, language, translator, tokenizer)

    if language != "english":
        print(f"\n{'='*60}")
        print("  TRANSLATED TEXT (English) — NLLB 1.3B")
        print(f"{'='*60}")
        print(english_text[:500] + ("…" if len(english_text) > 500 else ""))
        print(f"{'='*60}\n")

    # TTS
    print("[STAGE 3] Speaking multi-page …")
    t_tts_start = time.time()
    send_json(client_sock, {"cmd": "tts_started"})
    tts_module.speak(english_text)

    stopped = wait_for_tts_with_controls(tts_module, recv_queue, client_sock)
    tts_module.wait_until_done()
    send_json(client_sock, {"cmd": "tts_ended"})

    t_tts = time.time() - t_tts_start
    total_pipeline = time.time() - pipeline_t0

    print(f"\n{'─'*40}")
    print("  ⏱  TIMING BREAKDOWN (Multi-page)")
    print(f"{'─'*40}")
    print(f"  OCR (all)    : {t_ocr_total:.3f}s")
    print(f"  Translation  : {t_translate:.3f}s")
    print(f"  TTS Playback : {t_tts:.3f}s")
    print(f"{'─'*40}")
    print(f"  TOTAL        : {total_pipeline:.3f}s")
    print(f"{'─'*40}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║     FYDP SMART GLASSES — FINAL DEMO PIPELINE (BUTTONS)     ║
║          OCR → TRANSLATE (NLLB 1.3B) → SPEAK               ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── 1. Load TTS first (needed for welcome announcement) ──
    tts_module = load_tts()

    tts_module.speak("Welcome to F Y D P Glasses")
    tts_module.wait_until_done()

    # ── 2. Load remaining models ──
    print("\n[INFO] Loading all models (one-time cost) …\n")
    total_t0 = time.time()

    ocr_engine            = load_ocr_engine()
    translator, tokenizer = load_nllb_translator()
    unwarper              = load_unwarper()

    print(f"\n[INFO] All models loaded in {time.time()-total_t0:.2f}s")

    tts_module.speak("All models loaded")
    tts_module.wait_until_done()

    # ── 3. Wait for Pi to connect ──
    server_sock, client_sock = start_server(PORT)

    tts_module.speak("Smart glasses connected")
    tts_module.wait_until_done()

    # ── 4. Start receiver thread ──
    recv_queue = queue.Queue()
    recv_thread = threading.Thread(
        target=receiver_loop, args=(client_sock, recv_queue), daemon=True)
    recv_thread.start()

    # ── 5. Language selection ──
    tts_module.speak(
        "Choose language. "
        "Single press for Chinese. "
        "Double press for French. "
        "Triple press for Spanish."
    )
    tts_module.wait_until_done()

    send_json(client_sock, {"cmd": "choose_language"})
    msg = wait_for_json(recv_queue, "language")
    language = msg["value"]

    tts_module.speak(f"{language} selected")
    tts_module.wait_until_done()
    print(f"[CONFIG] Language: {language.upper()}")

    # ── 6. Book mode selection ──
    tts_module.speak(
        "Choose book mode. "
        "Single press for book mode. "
        "Double press for no book mode."
    )
    tts_module.wait_until_done()

    send_json(client_sock, {"cmd": "choose_book_mode"})
    msg = wait_for_json(recv_queue, "book_mode")
    book_mode = msg["value"] == "on"

    tts_module.speak(f"Book mode {'on' if book_mode else 'off'}")
    tts_module.wait_until_done()
    print(f"[CONFIG] Book mode: {'ON' if book_mode else 'OFF'}")

    # If English selected, skip translation
    if language == "english":
        translator, tokenizer = None, None

    # ── 7. Ready ──
    print("\n" + "=" * 60)
    print(f"  Language   : {language.upper()}")
    print(f"  Book Mode  : {'ON' if book_mode else 'OFF'}")
    print(f"  Translator : NLLB-200 Distilled 1.3B")
    print("=" * 60)

    tts_module.speak("Press button to start translating")
    tts_module.wait_until_done()

    # ── 8. Capture loop ──
    run_count = 0

    try:
        while True:
            send_json(client_sock, {"cmd": "wait_capture"})
            print("\n⏳ Waiting for capture from Pi …")

            # Wait for a response (single_capture or multi_capture_start)
            msg = wait_for_json(recv_queue)

            if msg.get("resp") == "single_capture":
                # Get the frame
                frame = wait_for_frame(recv_queue)
                if frame is None:
                    print("[WARN] Received empty frame, skipping")
                    continue
                run_count += 1
                process_and_speak(
                    frame, run_count, language, book_mode,
                    ocr_engine, unwarper, translator, tokenizer,
                    tts_module, recv_queue, client_sock)

            elif msg.get("resp") == "multi_capture_start":
                tts_module.speak("Multiple capture mode. Press button to capture pages. Hold to finish.")
                tts_module.wait_until_done()
                send_json(client_sock, {"cmd": "multi_capture_ready"})

                # Collect all frames until multi_capture_end
                frames = collect_multi_frames(recv_queue)
                if not frames:
                    tts_module.speak("No pages captured.")
                    tts_module.wait_until_done()
                    continue
                run_count += 1
                process_multi_and_speak(
                    frames, run_count, language, book_mode,
                    ocr_engine, unwarper, translator, tokenizer,
                    tts_module, recv_queue, client_sock)

            print("\n  Ready for next capture!\n")

    except ConnectionError:
        print("\n[ERROR] Pi disconnected.")
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user.")
    finally:
        tts_module.speak("Goodbye")
        tts_module.wait_until_done()
        try:
            client_sock.close()
        except Exception:
            pass
        try:
            server_sock.close()
        except Exception:
            pass
        print(f"\n[EXIT] Session ended. Total runs: {run_count}")


if __name__ == "__main__":
    main()
