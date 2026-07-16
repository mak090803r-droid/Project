"""
pipeline_cli_box3b1.py  (box3b + T5 Spell Correction)
======================================================
Copy of pipeline_cli_box3b.py with oliverguhr/spelling-correction-english-base
inserted as Stage 2.5 between Translation and TTS.

The T5-base model corrects OCR character-swap errors (intemal→internal,
siructural→structural) using sentence-level context on the English output.

Usage:
    python pipeline_cli_box3b1.py
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_project_dir(start_dir):
    """Find the repository's Project directory from nested working copies."""
    current = os.path.abspath(start_dir)
    for _ in range(10):
        if (os.path.isdir(os.path.join(current, "1Pipeline")) and
                os.path.isdir(os.path.join(current, "pics"))):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(f"Could not locate Project root above {start_dir}")


PROJECT_DIR  = _find_project_dir(PIPELINE_DIR)
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
OCR_LOCK = threading.Lock()


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


def beep_failure():
    """Distinct retry signal used when a saved frame fails post-capture QA."""
    _play_tone(700, 280, 0.32, volume=0.4)


GUIDANCE_PROMPTS = {
    "too_far": "Move closer.",
    "too_close": "Move back.",
    "bad_lighting": "Improve lighting.",
    "blurry": "Hold still.",
    "almost_there": "Almost ready.",
}

_GUIDANCE_CLIPS = {}
_GUIDANCE_CLIPS_LOCK = threading.Lock()


def prepare_guidance_clips(tts_module):
    """Synthesize short prompts once; playback never touches the TTS queue."""
    if _GUIDANCE_CLIPS:
        return
    with _GUIDANCE_CLIPS_LOCK:
        if _GUIDANCE_CLIPS:
            return
        synthesize = getattr(tts_module, "_synthesize", None)
        if synthesize is None:
            raise RuntimeError("pipertts._synthesize is unavailable")
        print("[AUDIO] Pre-synthesizing real-time guidance prompts …")
        for state, prompt in GUIDANCE_PROMPTS.items():
            chunks = synthesize(prompt)
            if not chunks:
                raise RuntimeError(f"Piper produced no audio for {prompt!r}")
            sample_rate = chunks[0].sample_rate
            channels = chunks[0].sample_channels
            arrays = []
            for chunk in chunks:
                audio = np.asarray(chunk.audio_int16_array, dtype=np.int16)
                arrays.append(
                    audio.reshape(-1, channels).astype(np.float32) / 32768.0)
            # A short lead-in wakes the DAC without delaying the instruction.
            lead_in = np.zeros(
                (int(sample_rate * 0.06), channels), dtype=np.float32)
            _GUIDANCE_CLIPS[state] = (
                np.concatenate([lead_in] + arrays, axis=0), sample_rate)
        print(f"[AUDIO] {len(_GUIDANCE_CLIPS)} spoken guidance clips ready.")


def _play_guidance_clip(state):
    import sounddevice as sd
    audio, sample_rate = _GUIDANCE_CLIPS[state]
    sd.play(audio, samplerate=sample_rate, blocking=True)


class AudioGuidance:
    """Latest-state spoken feedback, independent of the blocking TTS queue."""
    def __init__(self, tts_module, state_cooldown=2.0, global_cooldown=0.65):
        prepare_guidance_clips(tts_module)
        self._queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._last_by_state = {}
        self._last_any = 0.0
        self._state_cooldown = state_cooldown
        self._global_cooldown = global_cooldown
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def notify(self, state):
        if state not in GUIDANCE_PROMPTS or self._stop_event.is_set():
            return
        now = time.monotonic()
        if now - self._last_by_state.get(state, 0.0) < self._state_cooldown:
            return
        if now - self._last_any < self._global_cooldown:
            return
        self._last_by_state[state] = now
        self._last_any = now
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(state)
            print(f"  [GUIDANCE] {GUIDANCE_PROMPTS[state]}")
        except queue.Full:
            pass

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                state = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if state is None:
                break
            try:
                _play_guidance_clip(state)
            except Exception as exc:
                print(f"  [GUIDANCE] Playback failed: {exc}")

    def stop(self):
        self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
PREPROCESS_CONFIG = {
    # Winner of the 45-image / 225-prediction empirical sweep.
    "variant": "V_A_RAW_GRAYSCALE",
}


def preprocess_image(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return None
    # V_A: deliberately no blur, sharpening, CLAHE, gamma, or resizing.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # [V6 FIX] GaussianBlur removed — C525 plastic lens is already soft;
    # blurring on top of it merges thin character strokes and kills OCR on 11-12pt text.
    # gray     = cv2.GaussianBlur(gray, (3, 3), 0)


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

# PaddleOCR's language codes differ from NLLB's. PP-OCRv6 medium is a
# multilingual model, but this audited mapping selects the correct script.
PADDLE_OCR_LANG_MAP = {
    "french": "en",
    "chinese": "ch",
    "spanish": "en",
    "english": "en",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_ocr_engine(language):
    from paddleocr import PaddleOCR
    ocr_device = "gpu" if NLLB_DEVICE == "cuda" else "cpu"
    ocr_lang = PADDLE_OCR_LANG_MAP[language]
    print(f"[LOAD] OCR language audit: {language} -> PaddleOCR lang='{ocr_lang}'")
    print(f"[LOAD] Initializing PaddleOCR (onnxruntime, {ocr_device}) …")
    t = time.time()
    ocr = PaddleOCR(
        lang=ocr_lang,
        ocr_version="PP-OCRv6",
        use_doc_orientation_classify=False,
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
    prepare_guidance_clips(pipertts)
    print(f"[LOAD] TTS ready in {time.time()-t:.2f}s")
    return pipertts


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════
def save_frame(frame, run_count):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename  = f"capture_cli_{run_count:03d}_{timestamp}.jpg"
    filepath  = os.path.join(CAPTURED_DIR, filename)
    # Save as high-quality JPEG (95%) instead of slow PNG
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
    print("[STAGE 1] Running PaddleOCR …")
    ocr_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    with OCR_LOCK:
        result = ocr_engine.predict(ocr_input)
    lines = []
    for res in result:
        if "rec_texts" in res:
            for text in res["rec_texts"]:
                if text.strip():
                    lines.append(text.strip())
    extracted = " ".join(lines)

    for f in [os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")]:
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


# ══════════════════════════════════════════════════════════════════════════════
#  SPELL CORRECTION — oliverguhr/spelling-correction-english-base (T5-base)
# ══════════════════════════════════════════════════════════════════════════════
_SPELL_MODEL_NAME = "oliverguhr/spelling-correction-english-base"

def load_spell_corrector():
    """Load the T5-based spell correction model onto GPU."""
    t0 = time.time()
    print(f"[LOAD] Downloading/loading spell corrector: {_SPELL_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(_SPELL_MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(_SPELL_MODEL_NAME)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    model.eval()
    print(f"[LOAD] Spell corrector ready on {device} in {time.time()-t0:.2f}s")
    return {"model": model, "tokenizer": tokenizer, "device": device}


def run_spell_correction(corrector, text):
    """
    Correct OCR spelling errors using T5-base (oliverguhr).
    Splits text into sentences, corrects each individually for best context,
    then re-joins. Sentences longer than 128 tokens are chunked further.
    """
    if not text or not text.strip():
        return text

    model     = corrector["model"]
    tokenizer = corrector["tokenizer"]
    device    = corrector["device"]

    # Split into sentences (preserve paragraph structure)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    corrected_parts = []
    with torch.no_grad():
        for sentence in sentences:
            input_ids = tokenizer(
                sentence, return_tensors="pt", max_length=256,
                truncation=True, padding=False
            ).input_ids.to(device)

            outputs = model.generate(
                input_ids,
                max_length=256,
                num_beams=3,         # beam search for better quality
                early_stopping=True,
            )
            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_parts.append(corrected)

    return " ".join(corrected_parts)


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
#  FRAME QUALITY SCORING
# ══════════════════════════════════════════════════════════════════════════════
QUALITY_THRESHOLD = 50   # score floor (just to reject blurry motion)

def score_frame_quality(frame):
    """
    Score frame 0-100 based on:
      - Sharpness  (40 pts) — Laplacian variance (50 to 300 range, raised from 125 ceiling)
      - Brightness (30 pts) — mean pixel value, ideal 80-180
      - Evenness   (30 pts) — quadrant brightness consistency

    Returns: (score: int, details: dict)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Sharpness (40 pts) — Laplacian variance ──
    # [FIX 1] Raised ceiling from LapVar≥125 to LapVar≥300.
    # Old range let frames with LapVar=95-200 score 40/40 (fully sharp),
    # which allowed 7 noticeably soft images through the gate.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        sharpness_pts = 0
    elif laplacian_var >= 300:
        sharpness_pts = 40
    else:
        # Linear range: 50 → 300 maps to 0 → 40 pts
        sharpness_pts = int((laplacian_var - 50) / 250 * 40)

    # ── Brightness (30 pts) ──
    mean_brightness = float(np.mean(gray))
    if 80 <= mean_brightness <= 180:
        brightness_pts = 30
    elif mean_brightness < 80:
        brightness_pts = max(0, int(mean_brightness / 80 * 30))
    else:
        brightness_pts = max(0, int((255 - mean_brightness) / 75 * 30))

    # ── Evenness (30 pts) ──
    mid_h, mid_w = h // 2, w // 2
    quadrants = [
        gray[:mid_h, :mid_w],
        gray[:mid_h, mid_w:],
        gray[mid_h:, :mid_w],
        gray[mid_h:, mid_w:],
    ]
    quad_means = [float(np.mean(q)) for q in quadrants]
    quad_std   = float(np.std(quad_means))
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
#  TEXT REGION DETECTION & OUTER PAGE BOX GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def detect_text_region_size(frame, ocr_engine):
    """
    Run lightweight text detection on a downscaled frame to decide
    whether to say 'closer' or 'further', and to find the outer page bounding box.

    Returns: (hint, boxes_for_display, outer_box_info, num_lines)
      - boxes_for_display: list of 4-point polygons scaled back to original frame size.
      - outer_box_info: dict with 'rect' (x1,y1,x2,y2) and 'touching_edge' (bool).
      - num_lines: total number of detected text lines.
    """
    h_orig, w_orig = frame.shape[:2]

    # Downscale to 640px wide for reliable detection of small text (320 was too small)
    target_w = min(640, w_orig)
    scale = target_w / w_orig
    small = cv2.resize(frame, (target_w, int(h_orig * scale)))
    h_s, w_s = small.shape[:2]

    # Save to temp JPEG
    temp_det = os.path.join(PIPELINE_DIR, "_temp_det_preview.jpg")
    cv2.imwrite(temp_det, small, [cv2.IMWRITE_JPEG_QUALITY, 80])

    try:
        result = ocr_engine.predict(temp_det)
    except Exception:
        return None, [], None, 0
    finally:
        if os.path.exists(temp_det):
            os.remove(temp_det)

    # Extract box polygons
    boxes_small = []
    for res in result:
        if "dt_polys" in res:
            for poly in res["dt_polys"]:
                boxes_small.append(np.array(poly, dtype=np.float32))

    num_lines = len(boxes_small)
    if num_lines == 0:
        return "closer", [], None, 0

    # Scale boxes back to original resolution
    inv_scale = 1.0 / scale
    boxes_orig = []
    for poly in boxes_small:
        scaled_poly = (poly * inv_scale).astype(np.int32)
        boxes_orig.append(scaled_poly)

    # Compute outer bounding box enclosing ALL text lines
    all_x_min, all_y_min = w_orig, h_orig
    all_x_max, all_y_max = 0, 0
    touching_edges = 0
    # [FIX 4] Widened edge margin from 12*inv_scale (~36px) to 3.5% of frame width (~67px on 1920p).
    # This protects the first/last characters of text lines from JPEG chroma artifacts at the crop edge.
    edge_margin = int(0.035 * w_orig)

    for poly in boxes_orig:
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)

        all_x_min = min(all_x_min, x_min)
        all_y_min = min(all_y_min, y_min)
        all_x_max = max(all_x_max, x_max)
        all_y_max = max(all_y_max, y_max)

        if x_min < edge_margin or y_min < edge_margin or \
           x_max > w_orig - edge_margin or y_max > h_orig - edge_margin:
            touching_edges += 1

    # Check if outer page box touches screen boundaries (page cut off)
    page_touching_edge = (
        all_x_min < edge_margin or
        all_y_min < edge_margin or
        all_x_max > w_orig - edge_margin or
        all_y_max > h_orig - edge_margin
    )

    pad = 12
    outer_box_info = {
        "rect": (
            max(0, int(all_x_min) - pad),
            max(0, int(all_y_min) - pad),
            min(w_orig, int(all_x_max) + pad),
            min(h_orig, int(all_y_max) + pad),
        ),
        "touching_edge": page_touching_edge,
    }

    # Bounding area fraction
    frame_area = h_orig * w_orig
    total_box_area = (all_x_max - all_x_min) * (all_y_max - all_y_min)
    area_ratio = total_box_area / frame_area

    hint = None
    # Only guide to move closer if we see zero/very few lines
    if num_lines < 3:
        hint = "closer"
    # Guide to back up if outer page touches edge and fills almost entire camera frame
    elif page_touching_edge and area_ratio > 0.82:
        hint = "further"

    return hint, boxes_orig, outer_box_info, num_lines


# ══════════════════════════════════════════════════════════════════════════════
#  NON-BLOCKING DETECTION WORKER & THREAD CONTAINER
# ══════════════════════════════════════════════════════════════════════════════
class DetectionResult:
    """Thread-safe storage for background text detection results."""
    def __init__(self):
        self._lock = threading.Lock()
        self.hint = None
        self.boxes = []
        self.outer_box = None
        self.num_lines = 0
        self.running = False

    def update(self, hint, boxes, outer_box, num_lines):
        with self._lock:
            self.hint = hint
            self.boxes = boxes
            self.outer_box = outer_box
            self.num_lines = num_lines
            self.running = False

    def get(self):
        with self._lock:
            return self.hint, list(self.boxes), self.outer_box, self.num_lines

    def set_running(self):
        with self._lock:
            self.running = True

    def is_running(self):
        with self._lock:
            return self.running


def _detection_worker(frame, ocr_engine, det_result):
    try:
        hint, boxes, outer_box, num_lines = detect_text_region_size(frame, ocr_engine)
        det_result.update(hint, boxes, outer_box, num_lines)
    except Exception:
        det_result.update(None, [], None, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY OVERLAY ON PREVIEW (with outer page box)
# ══════════════════════════════════════════════════════════════════════════════
def draw_quality_overlay(frame, score, details, boxes, outer_box_info=None):
    """Draw quality bar + individual boxes + single outer page box on a copy."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Cyan boxes for text lines
    for poly in boxes:
        pts = poly.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Outer page boundary box
    if outer_box_info is not None:
        x1, y1, x2, y2 = outer_box_info["rect"]
        if outer_box_info["touching_edge"]:
            box_color = (0, 0, 255) # Red = page cut off at screen borders
            label = "PAGE CUT OFF"
        else:
            box_color = (0, 255, 0) # Green = page fully inside boundaries
            label = "PAGE OK"
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(display, label, (x1 + 5, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

    # Top overlay bar
    bar_h = 40
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    if score >= QUALITY_THRESHOLD:
        color = (0, 220, 0)     # green
    elif score >= 35:
        color = (0, 200, 255)   # yellow
    else:
        color = (0, 0, 220)     # red

    bar_w = int((score / 100) * (w - 20))
    cv2.rectangle(display, (10, 8), (10 + bar_w, bar_h - 8), color, -1)

    cv2.putText(display, f"Focus Score: {score}/100", (15, bar_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    detail_str = (f"Sharp:{details['sharpness_pts']}  "
                  f"Bright:{details['brightness_pts']}  "
                  f"Even:{details['evenness_pts']}")
    text_size = cv2.getTextSize(detail_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(display, detail_str, (w - text_size[0] - 15, bar_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return display


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-CAPTURE QUALITY LOOP
# ══════════════════════════════════════════════════════════════════════════════
def pre_capture_quality_loop(frame_holder, ocr_engine, tts_module,
                              key_queue, event_queue):
    """
    Quality-check loop that handles C525 autofocus hunting and premature triggers:
      1. Keeps stream smooth (~30 fps) using background thread detection.
      2. Requires 3 consecutive frames of good focus + framing (STABILITY HOLD).
      3. Requires >= 6 text lines (FULL PAGE GATE) to prevent trigger on single paragraphs.
    """
    print("\n[QUALITY] Entering advanced quality check …")
    print(f"[QUALITY] Requirements: Focus score ≥ {QUALITY_THRESHOLD} + Page box is GREEN + ≥ 6 text lines")
    print("[QUALITY] Hold still for 1 second once aligned. Press S to force-capture.")

    last_voice_time   = 0.0
    last_detect_time  = 0.0
    voice_cooldown    = 2.2      # seconds between voice guide reminders
    detect_interval   = 0.8      # [FIX 3] Reduced from 1.5s → 0.8s. Old: 3×1.5=4.5s min hold. New: 3×0.8=2.4s min hold.
    loop_sleep        = 0.03     # ~30 fps loop update

    # Stability hold tracker
    stability_counter = 0
    required_consecutive = 3     # must remain perfectly focused/aligned for 3 checks

    det_result = DetectionResult()

    while True:
        frame = frame_holder.get()
        if frame is None:
            time.sleep(0.1)
            continue

        # ── 1. Focus check (very fast, inline) ──
        score, details = score_frame_quality(frame)

        # ── 2. Run text detection in background (non-blocking) ──
        now = time.time()
        if now - last_detect_time >= detect_interval and not det_result.is_running():
            last_detect_time = now
            det_result.set_running()
            det_thread = threading.Thread(
                target=_detection_worker,
                args=(frame.copy(), ocr_engine, det_result),
                daemon=True)
            det_thread.start()

        # Get latest results
        cached_hint, cached_boxes, cached_outer, num_lines = det_result.get()

        # ── 3. Render preview display ──
        display = draw_quality_overlay(frame, score, details, cached_boxes, cached_outer)
        cv2.imshow("Smart Glasses Live Stream", cv2.resize(display, (640, 360)))
        cv2.waitKey(1)

        # ── 4. Voice hints (throttled) ──
        if cached_hint is not None and (now - last_voice_time >= voice_cooldown):
            last_voice_time = time.time()
            if cached_hint == "closer" and num_lines == 0:
                # Only tell to move closer if there's literally zero text visible
                tts_module.speak("move closer. move closer.")
                tts_module.wait_until_done()
            elif cached_hint == "further":
                tts_module.speak("back up. back up.")
                tts_module.wait_until_done()

        # ── 5. Stability & Alignment Gate ──
        page_ok = cached_outer is not None and not cached_outer["touching_edge"]

        # Conditions for a good capture:
        #   - Focus is sharp enough (score >= floor)
        #   - We see a full page block (at least 6 text lines)
        #   - The entire page fits on the screen (outer box is GREEN / not touching edges)
        #   - [FIX 2] Lighting is even across all quadrants (rejects half-shadowed captures)
        evenness_ok = details["evenness_std"] < 35
        frame_is_good = (score >= QUALITY_THRESHOLD) and (num_lines >= 6) and page_ok and evenness_ok

        if not evenness_ok and score >= QUALITY_THRESHOLD and page_ok:
            print(f"  [GATE] Rejected: uneven lighting (QuadStd={details['evenness_std']:.1f} >= 35)")

        if frame_is_good:
            stability_counter += 1
            print(f"  [STABLE HOLD] {stability_counter}/{required_consecutive} (Lines: {num_lines}, Focus: {score})")
        else:
            # Instantly reset if head shakes or text goes out of focus/cuts off
            if stability_counter > 0:
                print("  [STABLE HOLD] Reset! (Autofocus hunting or camera moved)")
            stability_counter = 0

        # Auto-capture triggers when held stable
        if stability_counter >= required_consecutive:
            print(f"\n[QUALITY] ✅ Stability hold achieved! Capturing frame.")
            beep_ready()
            return frame

        # ── 6. Keyboard / Pi override inputs ──
        try:
            while not key_queue.empty():
                key = key_queue.get_nowait()
                if key == 'q':
                    print("[QUALITY] ❌ Cancelled by user (Q)")
                    return None
                elif key == 's':
                    print(f"[QUALITY] ⚡ Force-capture triggered (S) at score {score}")
                    beep_ready()
                    return frame
        except queue.Empty:
            pass

        try:
            while not event_queue.empty():
                evt = event_queue.get_nowait()
                if evt.get("event") == "disconnect":
                    return None
                if evt.get("cmd") == "capture_from_pi":
                    print(f"[QUALITY] ⚡ Force-capture triggered from Pi (S) at score {score}")
                    beep_ready()
                    return frame
        except queue.Empty:
            pass

        time.sleep(loop_sleep)


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS & SPEAK
# ══════════════════════════════════════════════════════════════════════════════
def process_and_speak(frame, run_count, language, book_mode,
                      ocr_engine, unwarper, translator, tokenizer, tts_module,
                      spell_corrector=None,
                      key_queue=None, event_queue=None):
    print(f"\n{'─'*60}")
    print(f"  RUN #{run_count}")
    print(f"{'─'*60}\n")

    pipeline_t0 = time.time()
    beep_capture()
    img_path = save_frame(frame, run_count)

    # Score the actual JPEG that OCR would consume, not only the live frame.
    saved_frame = cv2.imread(img_path)
    if saved_frame is None:
        print("[POST-CAPTURE] Could not reload saved image; retry requested.")
        beep_failure()
        return False
    _, post_details = score_frame_quality(saved_frame)
    print(
        f"[POST-CAPTURE] LapVar={post_details['sharpness']:.1f}, "
        f"score components: sharp={post_details['sharpness_pts']}, "
        f"bright={post_details['brightness_pts']}, "
        f"even={post_details['evenness_pts']}")

    cv2.imshow("Captured Image (OCR Target)", cv2.resize(frame, (640, 360)))
    cv2.waitKey(1)

    # Stage 1: OCR
    extracted_text, t_ocr = run_ocr(ocr_engine, img_path, book_mode, unwarper)
    if not extracted_text:
        tone_failure()
        tts_module.speak("No text detected. Please try again.")
        tts_module.wait_until_done()
        print("[WARN] No text extracted. Skipping.\n")
        return False

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

    # Save OCR text to a comparison log file
    try:
        # PROJECT_DIR is defined at top of pipeline_cli_box3b.py
        log_dir = os.path.join(PROJECT_DIR, "1Pipeline", "final", "working")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "ocr_results_comparison.txt")
        with open(log_path, "a", encoding="utf-8") as f_log:
            f_log.write("="*80 + "\n")
            f_log.write(f"TIMESTAMP   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log.write(f"CODE VERSION: pipeline_cli_box3b1.py\n")
            f_log.write(f"RUN NUMBER  : {run_count}\n")
            f_log.write(f"IMAGE FILE  : {os.path.basename(img_path)}\n")
            f_log.write("-"*80 + "\n")
            f_log.write("EXTRACTED TEXT:\n")
            f_log.write(extracted_text + "\n")
            if language != "english":
                f_log.write("-"*80 + "\n")
                f_log.write("TRANSLATED TEXT:\n")
                f_log.write(english_text + "\n")
            f_log.write("="*80 + "\n\n")
    except Exception as e_log:
        print(f"[LOG ERROR] Failed to save OCR output to log: {e_log}")

    # Stage 2.5: T5 Spell Correction (on English text)
    t_spell = 0.0
    if spell_corrector is not None:
        print("[STAGE 2.5] Running T5 spell correction …")
        t_spell_start = time.time()
        english_text = run_spell_correction(spell_corrector, english_text)
        t_spell = time.time() - t_spell_start
        print(f"[STAGE 2.5] Spell correction complete in {t_spell:.3f}s")
        print(f"\n{'='*60}")
        print("  CORRECTED TEXT (T5 Spell)")
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
    print(f"  Spell Corr   : {t_spell:.3f}s")
    print(f"  TTS Playback : {t_tts:.3f}s")
    print(f"{'─'*40}")
    print(f"  TOTAL        : {total_pipeline:.3f}s")
    print(f"{'─'*40}")
    print(f"\n  Ready for next capture!\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  CLI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║  FYDP SMART GLASSES — BOX3B1 (T5 Spell Correction)         ║
║  OCR → TRANSLATE (NLLB 1.3B) → T5 SPELL FIX → SPEAK       ║
║                                                              ║
║  Controls:                                                   ║
║      S  =  enter quality check → auto-capture               ║
║      Q  =  quit                                              ║
║  Also works with S/Q on the Pi keyboard!                     ║
╚══════════════════════════════════════════════════════════════╝
"""


class SuppressOutput:
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

    print("[INFO] Initializing audio subsystem …")
    with SuppressOutput():
        tts_module = load_tts()

    tts_module.speak("Welcome to F Y D P Glasses")

    tts_module.wait_until_done()

    tts_module.speak("Choose language")
    language = ask_language()

    print("[INFO] Loading language-aware OCR and unwarping models (silently) …")
    with SuppressOutput():
        ocr_engine = load_ocr_engine(language)
        unwarper   = load_unwarper()

    tts_module.speak("Choose book mode")
    book_mode = ask_book_mode()

    if not book_mode:
        unwarper = None

    print(f"\n[INFO] Loading NLLB Translator for {language.upper()} …")
    with SuppressOutput():
        translator, tokenizer = load_nllb_translator(language)

    print("[INFO] Loading T5 Spell Corrector (oliverguhr/spelling-correction-english-base) …")
    spell_corrector = load_spell_corrector()

    tts_module.speak("All models loaded")
    tts_module.wait_until_done()

    server_sock, client_sock = start_server(PORT)
    tts_module.speak("Smart glasses connected. Press S to start translating.")
    tts_module.wait_until_done()

    frame_holder = FrameHolder()
    event_queue  = queue.Queue()
    recv_thread  = threading.Thread(
        target=receiver_loop,
        args=(client_sock, frame_holder, event_queue),
        daemon=True)
    recv_thread.start()

    key_queue = queue.Queue()
    start_keyboard_listener(key_queue)

    pipeline_busy = threading.Event()

    print("\n⏳ Waiting for capture … (S = capture, Q = quit)\n")

    run_count = 0

    try:
        while True:
            if not pipeline_busy.is_set():
                live_frame = frame_holder.get()
                if live_frame is not None:
                    cv2.imshow("Smart Glasses Live Stream", cv2.resize(live_frame, (640, 360)))
                    cv2.waitKey(1)

            try:
                event = event_queue.get_nowait()
                if event.get("event") == "disconnect" or event.get("cmd") == "quit":
                    print("\n[INFO] Pi disconnected.")
                    break
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
                                spell_corrector=spell_corrector,
                                key_queue=key_queue, event_queue=event_queue)
                        pipeline_busy.clear()
            except queue.Empty:
                pass

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
                                spell_corrector=spell_corrector,
                                key_queue=key_queue, event_queue=event_queue)
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
