"""
pipeline_cli_box3d.py  (Box3c + Paragraph-Aware Translation and TTS)
==================================================================================
Based on pipeline_cli_box.py, with software adjustments for the Logitech C525:

  1. UPSCALED OCR RESOLUTION: Upscales the 720p stream to 1.5x (1080p equivalent)
     before running PaddleOCR so small 12pt font is readable at a distance.
  2. STABILITY HOLD GATING: Requires 3 consecutive frames of good focus and
     framing before auto-capturing (prevents premature triggers).
  3. FULL PAGE GATE: Requires at least 6 text lines to be detected so it doesn't
     trigger on a single focused paragraph.
  4. OUTER BOUNDING BOX: Draws a single page bounding box (Green = OK, Red = Cut off).
  5. SMOOTH PREVIEW: Runs text detection in a background thread to prevent FPS drops.

Usage:
    C:/Users/ali/Desktop/FYDP/fydp/Scripts/python.exe pipeline_cli_box3d.py
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
import symspellpy
from symspellpy import SymSpell, Verbosity

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
PICS_DIR           = os.path.join(PROJECT_DIR, "pics")
CAPTURED_DIR       = os.path.join(PICS_DIR, "captured")
PARAGRAPH_TEST_DIR = os.path.join(PIPELINE_DIR, "paragraph_test_outputs")  # DEBUG — remove when done
os.makedirs(CAPTURED_DIR, exist_ok=True)
os.makedirs(PARAGRAPH_TEST_DIR, exist_ok=True)  # DEBUG — remove when done

PORT = 9999

# ══════════════════════════════════════════════════════════════════════════════
#  WIRE PROTOCOL — must match piweb_cli.py
# ══════════════════════════════════════════════════════════════════════════════
MSG_JSON  = 0x01
MSG_FRAME = 0x02

_send_lock = threading.Lock()
OCR_LOCK = threading.Lock()
SPELL_CORRECTOR = None


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


def _extract_ocr_lines(result, image_shape):
    """Return PaddleOCR text, confidence, and matching boxes as line records."""
    image_h, image_w = image_shape[:2]
    records = []

    for res in result:
        texts = list(res.get("rec_texts", []))
        scores = list(res.get("rec_scores", []))
        boxes = res.get("rec_boxes")
        polygons = res.get("rec_polys", res.get("dt_polys", []))

        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue

            bbox = None
            if boxes is not None and index < len(boxes):
                box = np.asarray(boxes[index]).reshape(-1)
                if box.size >= 4:
                    bbox = tuple(int(round(value)) for value in box[:4])
            if bbox is None and polygons is not None and index < len(polygons):
                polygon = np.asarray(polygons[index]).reshape(-1, 2)
                if polygon.size:
                    bbox = (
                        int(round(np.min(polygon[:, 0]))),
                        int(round(np.min(polygon[:, 1]))),
                        int(round(np.max(polygon[:, 0]))),
                        int(round(np.max(polygon[:, 1]))),
                    )
            if bbox is None:
                # Coordinate-free fallback keeps the original OCR flow usable.
                y1 = index * 24
                bbox = (0, y1, max(1, image_w), y1 + 20)

            x1, y1, x2, y2 = bbox
            x1 = max(0, min(image_w - 1, x1))
            y1 = max(0, min(image_h - 1, y1))
            x2 = max(x1 + 1, min(image_w, x2))
            y2 = max(y1 + 1, min(image_h, y2))
            score = float(scores[index]) if index < len(scores) else 0.0
            records.append({
                "text": text,
                "score": score,
                "bbox": (x1, y1, x2, y2),
                "width": x2 - x1,
                "height": y2 - y1,
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
            })

    return records


def merge_ocr_fragments_into_lines(records):
    """Merge PaddleOCR fragments that occupy the same printed text row."""
    if not records:
        return []

    median_height = max(
        8.0, float(np.median([record["height"] for record in records])))
    ordered = sorted(records, key=lambda record: (
        record["center_y"], record["bbox"][0]))
    rows = []

    for record in ordered:
        selected_row = None
        for row in reversed(rows[-4:]):
            center_difference = abs(record["center_y"] - row["center_y"])
            if center_difference > 0.30 * median_height:
                continue
            rx1, _, rx2, _ = record["bbox"]
            bx1, _, bx2, _ = row["bbox"]
            horizontal_gap = max(0, max(rx1, bx1) - min(rx2, bx2))
            if horizontal_gap <= 6.0 * median_height:
                selected_row = row
                break

        if selected_row is None:
            rows.append({
                "fragments": [record],
                "bbox": record["bbox"],
                "center_y": record["center_y"],
            })
            continue

        selected_row["fragments"].append(record)
        fragments = selected_row["fragments"]
        x1 = min(fragment["bbox"][0] for fragment in fragments)
        y1 = min(fragment["bbox"][1] for fragment in fragments)
        x2 = max(fragment["bbox"][2] for fragment in fragments)
        y2 = max(fragment["bbox"][3] for fragment in fragments)
        selected_row["bbox"] = (x1, y1, x2, y2)
        selected_row["center_y"] = float(np.mean([
            fragment["center_y"] for fragment in fragments]))

    merged = []
    for row in rows:
        fragments = sorted(
            row["fragments"], key=lambda fragment: fragment["bbox"][0])
        x1, y1, x2, y2 = row["bbox"]
        merged.append({
            "text": " ".join(fragment["text"] for fragment in fragments),
            "score": float(np.mean([
                fragment["score"] for fragment in fragments])),
            "bbox": (x1, y1, x2, y2),
            "width": x2 - x1,
            "height": y2 - y1,
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
        })
    return sorted(merged, key=lambda line: (
        line["center_y"], line["bbox"][0]))


def _horizontal_overlap_ratio(first, second):
    left = max(first[0], second[0])
    right = min(first[2], second[2])
    overlap = max(0, right - left)
    smaller_width = max(1, min(first[2] - first[0], second[2] - second[0]))
    return overlap / smaller_width


def group_ocr_lines_into_paragraphs(lines, image_shape):
    """
    Group OCR lines spatially while preserving deterministic reading order.

    Thresholds scale with the detected line height, so the behavior remains
    stable when the page is closer, farther away, or UVDoc changes resolution.
    """
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (
        line["center_y"], line["bbox"][0]))
    heights = [line["height"] for line in ordered if line["height"] > 0]
    median_height = float(np.median(heights)) if heights else 20.0
    median_height = max(8.0, median_height)

    comparable_gaps = []
    for previous, current in zip(ordered, ordered[1:]):
        if _horizontal_overlap_ratio(previous["bbox"], current["bbox"]) < 0.25:
            continue
        gap = current["bbox"][1] - previous["bbox"][3]
        if 0 <= gap <= 1.5 * median_height:
            comparable_gaps.append(gap)
    typical_gap = (
        float(np.median(comparable_gaps))
        if comparable_gaps else 0.35 * median_height)
    paragraph_gap_limit = min(
        max(0.80 * median_height, 2.20 * typical_gap),
        1.35 * median_height)
    heading_height = 1.25 * median_height

    def is_heading(line):
        text = line["text"].strip()
        return (
            bool(re.match(
                r"(?i)^(?:[\[\(\|]?\s*[0-9oO]{1,3}\s*[\]\)\|]?\s*)?"
                r"chapter\s+[ivxlcdm0-9]+", text))
            or bool(re.match(r"^\d+\.\d+\s+\S", text))
            or (
                line["height"] >= heading_height
                and len(text) <= 100
                and not text.rstrip().endswith((".", "?", "!"))
            )
        )

    def starts_new_paragraph(line):
        text = line["text"].strip()
        return bool(
            re.match(r"^[\[\(\|]\s*[0-9oO]{1,3}\s*[\]\)\|]", text)
            or re.match(r"^[0-9oO]{1,3}\]", text)
            or re.match(r"^\d+\.\d+\s+\S", text)
            or re.match(r"(?i)^chapter\s+[ivxlcdm0-9]+", text)
        )

    paragraph_lines = [[ordered[0]]]
    for current in ordered[1:]:
        previous = paragraph_lines[-1][-1]
        previous_box = previous["bbox"]
        current_box = current["bbox"]
        gap = current_box[1] - previous_box[3]
        overlap = _horizontal_overlap_ratio(previous_box, current_box)
        left_delta = abs(current_box[0] - previous_box[0])
        same_column = overlap >= 0.25 or left_delta <= 3.0 * median_height
        previous_heading = is_heading(previous)
        current_heading = is_heading(current)

        previous_is_chapter_heading = bool(re.search(
            r"(?i)\bchapter\s+[ivxlcdm0-9]+", previous["text"]))

        if (previous_is_chapter_heading and len(current["text"]) <= 120 and
                same_column and gap <= 1.20 * median_height):
            same_paragraph = True
        elif starts_new_paragraph(current):
            same_paragraph = False
        elif previous_heading and current_heading:
            same_paragraph = (
                same_column and gap <= 1.20 * median_height)
        elif previous_heading != current_heading:
            same_paragraph = False
        else:
            same_paragraph = same_column and gap <= paragraph_gap_limit
            if (same_paragraph and gap > 1.55 * typical_gap and
                    previous["text"].rstrip().endswith((".", "?", "!", ":"))):
                same_paragraph = False

        if same_paragraph:
            paragraph_lines[-1].append(current)
        else:
            paragraph_lines.append([current])

    image_h, image_w = image_shape[:2]
    padding = max(8, int(round(0.45 * median_height)))
    paragraphs = []
    for number, grouped_lines in enumerate(paragraph_lines, start=1):
        x1 = max(0, min(line["bbox"][0] for line in grouped_lines) - padding)
        y1 = max(0, min(line["bbox"][1] for line in grouped_lines) - padding)
        x2 = min(image_w, max(line["bbox"][2] for line in grouped_lines) + padding)
        y2 = min(image_h, max(line["bbox"][3] for line in grouped_lines) + padding)
        confidence_values = [line["score"] for line in grouped_lines]
        paragraphs.append({
            "number": number,
            "source_text": " ".join(line["text"] for line in grouped_lines),
            "bbox": (x1, y1, x2, y2),
            "lines": grouped_lines,
            "confidence": (
                float(np.mean(confidence_values)) if confidence_values else 0.0),
        })

    return paragraphs


def paragraph_page_location(paragraph, image_shape):
    """Describe a paragraph's vertical location for audio guidance."""
    image_h = max(1, image_shape[0])
    _, y1, _, y2 = paragraph["bbox"]
    relative_y = ((y1 + y2) / 2.0) / image_h
    if relative_y < 0.20:
        return "at the top of the page"
    if relative_y < 0.40:
        return "in the upper part of the page"
    if relative_y < 0.62:
        return "in the middle of the page"
    if relative_y < 0.82:
        return "in the lower part of the page"
    return "at the bottom of the page"


def draw_paragraph_overlay(image, paragraphs, active_index=None):
    """Draw numbered paragraph boxes, highlighting the paragraph being read."""
    canvas = image.copy()
    for index, paragraph in enumerate(paragraphs):
        x1, y1, x2, y2 = paragraph["bbox"]
        active = index == active_index
        color = (0, 220, 255) if active else (255, 120, 0)
        thickness = 4 if active else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"P{index + 1}"
        label_y = max(24, y1 - 8)
        cv2.putText(
            canvas, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, 2, cv2.LINE_AA)

    if active_index is not None and 0 <= active_index < len(paragraphs):
        status = f"READING PARAGRAPH {active_index + 1} OF {len(paragraphs)}"
    else:
        status = f"{len(paragraphs)} PARAGRAPHS DETECTED"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 620), 42), (20, 20, 20), -1)
    cv2.putText(
        canvas, status, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def show_paragraph_preview(image, paragraphs, active_index=None):
    """Display the OCR-coordinate image with paragraph mapping."""
    overlay = draw_paragraph_overlay(image, paragraphs, active_index)
    max_width, max_height = 1100, 800
    scale = min(
        1.0,
        max_width / max(1, overlay.shape[1]),
        max_height / max(1, overlay.shape[0]))
    if scale < 1.0:
        overlay = cv2.resize(
            overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Captured Image (OCR Target)", overlay)
    cv2.waitKey(1)


# DEBUG — remove when done (saves overlay image + OCR text log per run for inspection)
def save_run_debug_outputs(
        run_count, img_path, ocr_image, paragraphs,
        extracted_text, language, spell_changes):
    """
    Save a paragraph overlay image and a structured text log for this run.
    Files are written to PARAGRAPH_TEST_DIR so nothing pollutes the main log.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{run_count:03d}_{timestamp}"

    # ── 1. Paragraph overlay image ──────────────────────────────────────────
    if ocr_image is not None:
        overlay = draw_paragraph_overlay(ocr_image, paragraphs, active_index=None)
        overlay_path = os.path.join(
            PARAGRAPH_TEST_DIR, f"{base_name}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [DEBUG] Overlay saved  → {overlay_path}")

    # ── 2. Structured text log ───────────────────────────────────────────────
    log_path = os.path.join(PARAGRAPH_TEST_DIR, f"{base_name}_ocr.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as fout:
            fout.write("=" * 70 + "\n")
            fout.write(f"RUN          : {run_count}\n")
            fout.write(f"TIMESTAMP    : {timestamp}\n")
            fout.write(f"IMAGE FILE   : {os.path.basename(img_path)}\n")
            fout.write(f"LANGUAGE     : {language}\n")
            fout.write(f"PARAGRAPHS   : {len(paragraphs)}\n")
            fout.write("=" * 70 + "\n\n")

            fout.write("── RAW EXTRACTED TEXT ──\n")
            fout.write(extracted_text + "\n\n")

            fout.write("── PARAGRAPH DETAIL ──\n")
            for paragraph in paragraphs:
                fout.write("-" * 60 + "\n")
                fout.write(
                    f"P{paragraph['number']} | "
                    f"lines={len(paragraph['lines'])} | "
                    f"conf={paragraph['confidence']:.3f} | "
                    f"bbox={paragraph['bbox']}\n")
                fout.write(f"  SOURCE    : {paragraph['source_text']}\n")
                if language != "english" and "translated_text" in paragraph:
                    fout.write(
                        f"  TRANSLATED: {paragraph['translated_text']}\n")
                if "spoken_text" in paragraph:
                    fout.write(
                        f"  SPOKEN    : {paragraph['spoken_text']}\n")
                if paragraph.get("spell_changes"):
                    fout.write(
                        f"  SPELL FIXES: {paragraph['spell_changes']}\n")

            if spell_changes:
                fout.write("\n── SPELL CORRECTIONS SUMMARY ──\n")
                for para_num, original, replacement in spell_changes:
                    fout.write(f"  P{para_num}: {original!r} → {replacement!r}\n")

            fout.write("\n" + "=" * 70 + "\n")
        print(f"  [DEBUG] OCR log saved  → {log_path}")
    except Exception as save_err:
        print(f"  [DEBUG] Could not save OCR log: {save_err}")


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
        return "", [], None, time.time() - t0
    print("[STAGE 1] Running PaddleOCR …")
    ocr_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    with OCR_LOCK:
        result = ocr_engine.predict(ocr_input)
    raw_lines = _extract_ocr_lines(result, ocr_input.shape)
    lines = merge_ocr_fragments_into_lines(raw_lines)
    paragraphs = group_ocr_lines_into_paragraphs(lines, ocr_input.shape)
    extracted = " ".join(
        paragraph["source_text"] for paragraph in paragraphs)

    for f in [os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")]:
        if os.path.exists(f):
            os.remove(f)

    elapsed = time.time() - t0
    print(
        f"[STAGE 1] OCR complete in {elapsed:.3f}s  "
        f"({len(raw_lines)} detections -> {len(lines)} lines, "
        f"{len(paragraphs)} paragraphs)")
    return extracted, paragraphs, ocr_input, elapsed


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


def _split_translation_units(text, src_lang_code):
    """Split a paragraph at sentence boundaries without losing punctuation."""
    if src_lang_code == "zh":
        units = re.split(r"(?<=[。？！!?])", text)
    else:
        units = re.split(r"(?<=[.!?])\s+", text)
    return [unit.strip() for unit in units if unit.strip()]


def _fit_translation_unit(unit, tokenizer, src_lang_code, max_tokens=400):
    """Keep every NLLB request safely below its context limit."""
    if len(tokenizer.encode(unit)) <= max_tokens:
        return [unit]

    is_chinese = src_lang_code == "zh"
    pieces = list(unit) if is_chinese else unit.split()
    joiner = "" if is_chinese else " "
    chunks = []
    current = []
    for piece in pieces:
        candidate = joiner.join(current + [piece])
        if current and len(tokenizer.encode(candidate)) > max_tokens:
            chunks.append(joiner.join(current).strip())
            current = [piece]
        else:
            current.append(piece)
    if current:
        chunks.append(joiner.join(current).strip())
    return [chunk for chunk in chunks if chunk]


def run_translation_nllb_paragraphs(
        paragraphs, language, translator, tokenizer):
    """
    Translate every paragraph in one efficient NLLB batch while preserving IDs.

    Paragraphs may be split into sentence/token-safe chunks internally, but all
    translated chunks are reassembled into their original paragraph before TTS.
    """
    translated_paragraphs = [dict(paragraph) for paragraph in paragraphs]
    if translator is None or tokenizer is None:
        print("[STAGE 2] Skipping translation (English source)")
        for paragraph in translated_paragraphs:
            paragraph["translated_text"] = paragraph["source_text"]
        return translated_paragraphs, 0.0

    print(
        f"[STAGE 2] Translating {len(paragraphs)} paragraph(s) -> English "
        f"in one NLLB batch …")
    t0 = time.time()
    src_lang_code = PIPELINE_LANG_TO_NLLB[language]
    src_token = NLLB_LANG_MAP[src_lang_code]
    tgt_token = NLLB_LANG_MAP["en"]
    tokenizer.src_lang = src_token

    flat_chunks = []
    chunk_owners = []
    for paragraph_index, paragraph in enumerate(translated_paragraphs):
        sentence_units = _split_translation_units(
            paragraph["source_text"], src_lang_code)
        if not sentence_units:
            sentence_units = [paragraph["source_text"]]
        for unit in sentence_units:
            for chunk in _fit_translation_unit(
                    unit, tokenizer, src_lang_code):
                flat_chunks.append(chunk)
                chunk_owners.append(paragraph_index)

    if not flat_chunks:
        for paragraph in translated_paragraphs:
            paragraph["translated_text"] = paragraph["source_text"]
        return translated_paragraphs, time.time() - t0

    tokenized_batch = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk))
        for chunk in flat_chunks
    ]
    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    results = translator.translate_batch(
        tokenized_batch, target_prefix=target_prefixes)

    translated_by_paragraph = [
        [] for _ in translated_paragraphs]
    for owner, result in zip(chunk_owners, results):
        output_tokens = result.hypotheses[0]
        raw_decoded = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens))
        clean_text = raw_decoded.replace(tgt_token, "").strip()
        if clean_text:
            translated_by_paragraph[owner].append(clean_text)

    for index, paragraph in enumerate(translated_paragraphs):
        translated_text = " ".join(translated_by_paragraph[index]).strip()
        paragraph["translated_text"] = (
            translated_text if translated_text else paragraph["source_text"])

    elapsed = time.time() - t0
    print(
        f"[STAGE 2] Paragraph translation complete in {elapsed:.3f}s "
        f"({len(flat_chunks)} translation chunks)")
    return translated_paragraphs, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSERVATIVE OCR SPELL CORRECTION — SymSpell candidate generator
# ═══════════════════════════════════════════════════════════════════════════════

# These terms are always preserved. Generic rules below also preserve every
# acronym, mixed-case identifier, number, hyphenated term, and bracketed span.
_SPELL_PROTECTED_TERMS = {
    "airworthiness", "aerodynamic", "avionics", "barometer", "barometers",
    "bidirectional", "brushless", "chipset", "chipsets", "dynamometer",
    "electromagnetic", "fuselage", "maneuvering", "manoeuvring", "multirotor",
    "odometry", "powertrain", "telemetry", "topologically", "transmitter",
    "transmitters", "unwarping", "voxelization",
}

_BRACKETED_SPAN_RE = re.compile(
    r"(\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>)"
)
_PLAIN_TOKEN_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")

# A correction must be a frequent dictionary word and a visually plausible
# OCR edit. These thresholds intentionally prefer a missed correction over a
# false replacement that changes the document's meaning.
_SPELL_MIN_WORD_LENGTH = 6
_SPELL_MIN_CANDIDATE_COUNT = 300_000
_SPELL_MIN_DOMINANCE_RATIO = 100


def load_spell_corrector():
    """Load SymSpell as a candidate finder; compound rewriting is disabled."""
    t0 = time.time()
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = os.path.join(
        os.path.dirname(symspellpy.__file__),
        "frequency_dictionary_en_82_765.txt")
    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        raise RuntimeError(f"Could not load SymSpell dictionary: {dictionary_path}")
    print(f"[LOAD] Conservative SymSpell dictionary loaded in {time.time()-t0:.2f}s")
    return sym_spell


def _is_abbreviation_or_identifier(word):
    """Return True for acronyms, proper names, and mixed-case identifiers."""
    if len(word) <= 4:
        return True
    if word[0].isupper():
        return True
    if any(char.isupper() for char in word[1:]):
        return True
    if re.fullmatch(r"[A-Z]{2,}[a-z]?", word):
        return True
    return False


def _is_plausible_ocr_edit(source, candidate):
    """Allow only a small set of common visual OCR confusions."""
    source = source.lower()
    candidate = candidate.lower()
    if source == candidate:
        return False

    # Multi-character glyph confusions such as "intemal" -> "internal".
    def glyph_key(value):
        return value.replace("rn", "m").replace("cl", "d").replace("vv", "w")

    if glyph_key(source) == glyph_key(candidate):
        return True

    # One visually similar character substitution.
    if len(source) == len(candidate):
        differences = [
            (left, right)
            for left, right in zip(source, candidate)
            if left != right
        ]
        if len(differences) == 1:
            allowed_groups = (
                frozenset(("i", "l")),
                frozenset(("i", "j")),
                frozenset(("c", "e")),
                frozenset(("u", "v")),
                frozenset(("n", "r")),
                frozenset(("t", "f")),
            )
            return frozenset(differences[0]) in allowed_groups

    return False


def _correct_plain_segment(sym_spell, segment, changes):
    """Correct unprotected whitespace-delimited tokens without reformatting."""
    pieces = re.split(r"(\s+)", segment)
    for index, token in enumerate(pieces):
        if not token or token.isspace():
            continue

        # Never touch partial/malformed brackets, codes, measurements,
        # hyphenated technical terms (Li-Ion/Li-on), or slash-separated terms.
        if any(char in token for char in "[](){}<>0123456789-/\\+_=|@#%"):
            continue

        match = _PLAIN_TOKEN_RE.fullmatch(token)
        if match is None:
            continue
        prefix, word, suffix = match.groups()
        lower_word = word.lower()

        if len(word) < _SPELL_MIN_WORD_LENGTH:
            continue
        if _is_abbreviation_or_identifier(word):
            continue
        if lower_word in _SPELL_PROTECTED_TERMS:
            continue

        # A valid dictionary word is never rewritten. This prevents contextual
        # false positives such as "radio" -> "audio" or "array" -> "army".
        if lower_word in sym_spell.words:
            continue

        suggestions = sym_spell.lookup(
            lower_word,
            Verbosity.ALL,
            max_edit_distance=2,
            include_unknown=False)
        eligible = [
            suggestion for suggestion in suggestions
            if suggestion.distance in (1, 2)
            and suggestion.term.isalpha()
            and suggestion.count >= _SPELL_MIN_CANDIDATE_COUNT
            and _is_plausible_ocr_edit(lower_word, suggestion.term)
        ]
        if not eligible:
            continue

        eligible.sort(key=lambda suggestion: suggestion.count, reverse=True)
        best = eligible[0]
        if (len(eligible) > 1 and
                best.count < eligible[1].count * _SPELL_MIN_DOMINANCE_RATIO):
            continue

        replacement = best.term
        pieces[index] = f"{prefix}{replacement}{suffix}"
        changes.append((word, replacement))

    return "".join(pieces)


def run_spell_correction(sym_spell, text):
    """
    Apply only high-confidence OCR corrections.

    Text inside brackets is copied byte-for-byte. Abbreviations, identifiers,
    numbers, technical codes, and hyphenated words are never passed to
    SymSpell. Returns both the corrected text and an auditable change list.
    """
    if sym_spell is None or not text or not text.strip():
        return text, []

    changes = []
    chunks = _BRACKETED_SPAN_RE.split(text)
    corrected_chunks = []
    for chunk in chunks:
        if not chunk:
            continue
        if _BRACKETED_SPAN_RE.fullmatch(chunk):
            corrected_chunks.append(chunk)
        else:
            corrected_chunks.append(
                _correct_plain_segment(sym_spell, chunk, changes))
    return "".join(corrected_chunks), changes


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


def _clear_queue_safely(target_queue):
    if target_queue is None:
        return
    try:
        while not target_queue.empty():
            target_queue.get_nowait()
    except queue.Empty:
        pass


def _speak_segment_with_stop(
        tts_module, text, key_queue=None, event_queue=None):
    """Speak one paragraph segment; return True if the user stops playback."""
    tts_module.speak(text)
    stopped = False
    while tts_module.is_speaking():
        if key_queue is not None and not key_queue.empty():
            _clear_queue_safely(key_queue)
            tts_module.stop()
            stopped = True
            break

        if event_queue is not None and not event_queue.empty():
            stop_signal = False
            try:
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    if (event.get("cmd") == "capture_from_pi" or
                            event.get("event") == "disconnect"):
                        stop_signal = True
            except queue.Empty:
                pass
            if stop_signal:
                tts_module.stop()
                stopped = True
                break
        time.sleep(0.05)

    tts_module.wait_until_done()
    return stopped


def run_tts_paragraphs(
        tts_module, paragraphs, ocr_image,
        key_queue=None, event_queue=None):
    """Announce, highlight, and speak translated text paragraph by paragraph."""
    readable = [
        paragraph for paragraph in paragraphs
        if paragraph.get("spoken_text", "").strip()
    ]
    if not readable:
        return 0.0

    print(
        f"[STAGE 3] Speaking {len(readable)} paragraph(s) in reading order "
        f"… (press any key to stop)")
    t0 = time.time()
    _clear_queue_safely(key_queue)
    _clear_queue_safely(event_queue)
    stopped = False

    for index, paragraph in enumerate(readable):
        show_paragraph_preview(ocr_image, readable, active_index=index)
        location = paragraph_page_location(paragraph, ocr_image.shape)
        announcement = (
            f"Paragraph {index + 1} of {len(readable)}, {location}.")
        spoken_segment = f"{announcement} {paragraph['spoken_text'].strip()}"
        print(
            f"[STAGE 3] Paragraph {index + 1}/{len(readable)} — "
            f"{location}")
        if _speak_segment_with_stop(
                tts_module, spoken_segment, key_queue, event_queue):
            stopped = True
            break

    show_paragraph_preview(ocr_image, readable, active_index=None)
    _clear_queue_safely(key_queue)
    _clear_queue_safely(event_queue)
    elapsed = time.time() - t0
    if stopped:
        print(f"[STAGE 3] Paragraph TTS stopped after {elapsed:.3f}s")
    else:
        print(f"[STAGE 3] Paragraph TTS finished in {elapsed:.3f}s")
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

    # Stage 1: OCR with line coordinates and paragraph grouping.
    extracted_text, paragraphs, ocr_image, t_ocr = run_ocr(
        ocr_engine, img_path, book_mode, unwarper)
    if not extracted_text or not paragraphs:
        tone_failure()
        tts_module.speak("No text detected. Please try again.")
        tts_module.wait_until_done()
        print("[WARN] No text extracted. Skipping.\n")
        return False

    tone_success()
    if ocr_image is None:
        ocr_image = saved_frame
    show_paragraph_preview(ocr_image, paragraphs, active_index=None)

    print(f"\n{'='*60}")
    print("  EXTRACTED TEXT")
    print(f"{'='*60}")
    print(extracted_text)
    print(f"{'='*60}\n")

    print(f"[PARAGRAPHS] Detected {len(paragraphs)} paragraph(s):")
    for paragraph in paragraphs:
        location = paragraph_page_location(paragraph, ocr_image.shape)
        print(
            f"  P{paragraph['number']} | {len(paragraph['lines'])} line(s) | "
            f"confidence={paragraph['confidence']:.3f} | {location}")
        print(f"    {paragraph['source_text']}")

    # Stage 2: translate all paragraph chunks in one GPU batch, retaining IDs.
    translated_paragraphs, t_translate = run_translation_nllb_paragraphs(
        paragraphs, language, translator, tokenizer)
    english_text = " ".join(
        paragraph["translated_text"] for paragraph in translated_paragraphs)

    if language != "english":
        print(f"\n{'='*60}")
        print("  PARAGRAPH TRANSLATIONS (English) — NLLB 1.3B")
        print(f"{'='*60}")
        for paragraph in translated_paragraphs:
            print(
                f"[Paragraph {paragraph['number']}] "
                f"{paragraph['translated_text']}")
        print(f"{'='*60}\n")

    # Stage 2.5: apply the existing conservative SymSpell rules separately.
    t_spell = 0.0
    all_spell_changes = []
    spell_t0 = time.time()
    for paragraph in translated_paragraphs:
        translated_text = paragraph["translated_text"]
        if SPELL_CORRECTOR is not None:
            corrected_text, changes = run_spell_correction(
                SPELL_CORRECTOR, translated_text)
        else:
            corrected_text, changes = translated_text, []
        paragraph["spoken_text"] = corrected_text
        paragraph["spell_changes"] = changes
        for original, replacement in changes:
            all_spell_changes.append(
                (paragraph["number"], original, replacement))
    t_spell = time.time() - spell_t0 if SPELL_CORRECTOR is not None else 0.0
    corrected_text = " ".join(
        paragraph["spoken_text"] for paragraph in translated_paragraphs)

    if SPELL_CORRECTOR is not None:
        if all_spell_changes:
            print(
                f"[STAGE 2.5] Applied {len(all_spell_changes)} conservative "
                f"correction(s) in {t_spell:.3f}s:")
            for paragraph_number, original, replacement in all_spell_changes:
                print(
                    f"  P{paragraph_number}: {original} -> {replacement}")
        else:
            print(
                f"[STAGE 2.5] No sufficiently confident corrections; "
                f"original preserved ({t_spell:.3f}s)")

    # DEBUG — remove when done
    save_run_debug_outputs(
        run_count, img_path, ocr_image, translated_paragraphs,
        extracted_text, language, all_spell_changes)

    # Save both full-page text and the source-to-audio paragraph mapping.
    try:
        log_dir = os.path.join(PROJECT_DIR, "1Pipeline", "final", "working")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "ocr_results_comparison.txt")
        with open(log_path, "a", encoding="utf-8") as f_log:
            f_log.write("="*80 + "\n")
            f_log.write(f"TIMESTAMP   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log.write("CODE VERSION: pipeline_cli_box3d.py\n")
            f_log.write(f"RUN NUMBER  : {run_count}\n")
            f_log.write(f"IMAGE FILE  : {os.path.basename(img_path)}\n")
            f_log.write(f"PARAGRAPHS  : {len(translated_paragraphs)}\n")
            f_log.write("-"*80 + "\n")
            f_log.write("EXTRACTED TEXT:\n")
            f_log.write(extracted_text + "\n")
            if language != "english":
                f_log.write("-"*80 + "\n")
                f_log.write("TRANSLATED TEXT:\n")
                f_log.write(english_text + "\n")
            if all_spell_changes:
                f_log.write("-"*80 + "\n")
                f_log.write("CONSERVATIVELY CORRECTED TEXT:\n")
                f_log.write(corrected_text + "\n")

            for paragraph in translated_paragraphs:
                f_log.write("-"*80 + "\n")
                f_log.write(
                    f"PARAGRAPH {paragraph['number']} | "
                    f"BBOX={paragraph['bbox']} | "
                    f"CONFIDENCE={paragraph['confidence']:.3f}\n")
                f_log.write(
                    f"SOURCE     : {paragraph['source_text']}\n")
                if language != "english":
                    f_log.write(
                        f"TRANSLATED : {paragraph['translated_text']}\n")
                f_log.write(
                    f"SPOKEN     : {paragraph['spoken_text']}\n")
                if paragraph["spell_changes"]:
                    f_log.write(
                        f"CORRECTIONS: {paragraph['spell_changes']}\n")
            f_log.write("="*80 + "\n\n")
    except Exception as log_error:
        print(f"[LOG ERROR] Failed to save OCR output to log: {log_error}")

    # Stage 3: each spoken paragraph remains tied to its highlighted source box.
    t_tts = run_tts_paragraphs(
        tts_module, translated_paragraphs, ocr_image,
        key_queue, event_queue)

    total_pipeline = time.time() - pipeline_t0
    print(f"\n{'─'*40}")
    print("  ⏱  TIMING BREAKDOWN")
    print(f"{'─'*40}")
    print(f"  OCR          : {t_ocr:.3f}s")
    if language != "english":
        print(f"  Translation  : {t_translate:.3f}s")
    else:
        print("  Translation  : skipped")
    print(f"  Spell Corr   : {t_spell:.3f}s")
    print(f"  TTS Playback : {t_tts:.3f}s")
    print(f"{'─'*40}")
    print(f"  TOTAL        : {total_pipeline:.3f}s")
    print(f"{'─'*40}")
    print("\n  Ready for next capture!\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  CLI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   FYDP SMART GLASSES — BOX3D PARAGRAPH-AWARE PIPELINE      ║
║          OCR → TRANSLATE (NLLB 1.3B) → SPEAK               ║
║                                                              ║
║  Controls:                                                   ║
║      S  =  enter quality check → auto-capture               ║
║      Q  =  quit                                              ║
║  Also works with S/Q on the Pi keyboard!                     ║
║                                                              ║
║  Quality Feedback v3:                                        ║
║      • Outer PAGE box: green = OK, red = cut off             ║
║      • Focus Score (target ≥ 50)                             ║
║      • Requires ≥ 6 lines to ensure full page captured       ║
║      • Stability Hold: requires holding still for 1s         ║
║      • Upscales 720p stream to 1.5x (1080p equivalent)       ║
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
    global SPELL_CORRECTOR
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

    print("[INFO] Loading conservative SymSpell dictionary …")
    try:
        SPELL_CORRECTOR = load_spell_corrector()
    except Exception as spell_error:
        SPELL_CORRECTOR = None
        print(
            f"[WARN] SymSpell unavailable; continuing with original text: "
            f"{spell_error}")

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
                                key_queue, event_queue)
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
