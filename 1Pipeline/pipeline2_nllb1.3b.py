"""
pipeline2_nllb1.3b.py
=====================
Integrated LIVE CAMERA → OCR → Translation → TTS pipeline.
Image acquisition from Raspberry Pi smart glasses via TCP socket stream.
Uses NLLB-200 Distilled 1.3B (via CTranslate2) for translation.

Usage:
    python pipeline2_nllb1.3b.py

On launch you pick:
    1. Source language  (French / Chinese / English-only)
    2. Book mode        (curved-page unwarping via UVDoc)

Camera Acquisition:
    - Listens on TCP port 9999 for the Pi camera stream
    - Keeps waiting indefinitely until the Pi connects (no error on startup)
    - Live preview window shows the feed in real-time

Controls (in the live preview window):
    S  ->  Capture current frame, save it, and run the full pipeline
    Q  ->  Quit

All heavy models (OCR, Translator, TTS) are loaded ONCE at startup.
"""

import os
import sys
import re
import time
import threading
import socket
import struct
import pickle
import cv2
import numpy as np
import torch
import ctranslate2
import transformers

# ======================================================================
#  PATHS
# ======================================================================
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# Captured frames saved here (sibling "pics" folder, same as comms check rx)
PICS_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, "..", "pics"))
os.makedirs(PICS_DIR, exist_ok=True)

# ======================================================================
#  IMAGE PREPROCESSING
# ======================================================================
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

_S_VAL = PREPROCESS_CONFIG["sharpen_strength"]
_SHARPEN_KERNEL = np.array([
    [-_S_VAL,         -_S_VAL,   -_S_VAL],
    [-_S_VAL, 1+8*_S_VAL,        -_S_VAL],
    [-_S_VAL,         -_S_VAL,   -_S_VAL],
])


def unsharp_mask(image, sigma=1.0, strength=1.5):
    """Apply unsharp-mask sharpening for maximum OCR readability."""
    blurred   = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def preprocess_frame(frame):
    """
    Grayscale preprocessing on a live frame (numpy BGR array):
    denoise -> CLAHE -> gamma -> optional sharpen -> optional upscale.
    Returns the processed grayscale image ready for OCR.
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


def save_frame_for_ocr(frame):
    """
    Sharpen the full-resolution BGR frame, save as PNG, return the file path.
    File is saved to PICS_DIR with a timestamp name.
    """
    sharpened = unsharp_mask(frame, sigma=1.0, strength=1.5)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename  = f"capture_{timestamp}.png"
    filepath  = os.path.join(PICS_DIR, filename)
    cv2.imwrite(filepath, sharpened)
    print(f"  Saved capture ({frame.shape[1]}x{frame.shape[0]}) -> {filepath}")
    return filepath


def preprocess_image_from_path(img_path):
    """Preprocess an image from a disk path (used after saving the captured frame)."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return None
    return preprocess_frame(img)


# ======================================================================
#  NLLB 1.3B  CONFIGURATION
# ======================================================================
if torch.cuda.is_available():
    NLLB_DEVICE       = "cuda"
    NLLB_COMPUTE_TYPE = "float16"
    os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
    print("[INFO] Hardware Target: NVIDIA GPU (CUDA Optimization Active)")
else:
    NLLB_DEVICE       = "cpu"
    NLLB_COMPUTE_TYPE = "int8"
    print("[INFO] Hardware Target: CPU ONLY (Fallback Int8 Execution Active)")

NLLB_MODEL_DIR  = os.path.join(os.path.dirname(PIPELINE_DIR), "nllb-200-1.3B-ct2")
NLLB_BASE_MODEL = "facebook/nllb-200-distilled-1.3B"

NLLB_LANG_MAP = {
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "en": "eng_Latn"
}

PIPELINE_LANG_TO_NLLB = {
    "french":  "fr",
    "chinese": "zh",
}


# ======================================================================
#  MODEL LOADERS
# ======================================================================
def load_ocr_engine():
    """Load PaddleOCR engine (done once). Uses ONNX Runtime backend for GPU without pybind11 conflicts."""
    from paddleocr import PaddleOCR
    ocr_device = "gpu" if NLLB_DEVICE == "cuda" else "cpu"
    print(f"[LOAD] Initializing PaddleOCR engine (engine=onnxruntime, device={ocr_device}) ...")
    t   = time.time()
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
    """Load UVDoc text-image unwarper for book mode."""
    from paddleocr import TextImageUnwarping
    print("[LOAD] Initializing UVDoc unwarper (book mode) ...")
    t        = time.time()
    unwarper = TextImageUnwarping(model_name="UVDoc", engine="paddle")
    print(f"[LOAD] UVDoc ready in {time.time()-t:.2f}s")
    return unwarper


def load_nllb_translator(language):
    """
    Load NLLB-200 1.3B Distilled model via CTranslate2.
    Returns (translator, tokenizer) or (None, None) for English.
    """
    if language == "english":
        return None, None

    print(f"[LOAD] Loading NLLB-200 1.3B Distilled translator ...")
    t = time.time()

    if not os.path.exists(NLLB_MODEL_DIR):
        print(f"[INFO] Compiling {NLLB_BASE_MODEL} into CTranslate2 binary ...")
        start_conv    = time.time()
        conversion_cmd = (
            f"ct2-transformers-converter --model {NLLB_BASE_MODEL} "
            f"--output_dir {NLLB_MODEL_DIR} --force --quantization {NLLB_COMPUTE_TYPE}"
        )
        exit_code = os.system(conversion_cmd)
        if exit_code != 0:
            raise RuntimeError("[ERROR] Model compilation failed.")
        print(f"[INFO] Saved to '{NLLB_MODEL_DIR}' ({time.time()-start_conv:.2f}s)")

    translator = ctranslate2.Translator(NLLB_MODEL_DIR, device=NLLB_DEVICE, compute_type=NLLB_COMPUTE_TYPE)
    tokenizer  = transformers.AutoTokenizer.from_pretrained(NLLB_BASE_MODEL)

    print(f"[LOAD] NLLB 1.3B ready in {time.time()-t:.2f}s")
    return translator, tokenizer


def load_tts():
    """Pre-load Piper TTS voice model."""
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)

    import pipertts
    print("[LOAD] Pre-loading Piper TTS voice ...")
    t = time.time()
    pipertts.ensure_model()
    pipertts.preload()
    print(f"[LOAD] TTS ready in {time.time()-t:.2f}s")
    return pipertts


# ======================================================================
#  PIPELINE STAGES
# ======================================================================
def run_ocr(ocr_engine, img_path, book_mode, unwarper):
    """Run OCR on a saved image file. Returns (extracted_text, elapsed_seconds)."""
    t0 = time.time()

    if book_mode and unwarper is not None:
        print("[STAGE 1] Unwarping curved page ...")
        unwarp_result = unwarper.predict(img_path, batch_size=1)
        for res in unwarp_result:
            unwarped_img = res["doctr_img"]
        temp_path = os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")
        cv2.imwrite(temp_path, unwarped_img)
        img_path  = temp_path

    print("[STAGE 1] Preprocessing image ...")
    processed = preprocess_image_from_path(img_path)
    if processed is None:
        return "", time.time() - t0

    temp_proc = os.path.join(PIPELINE_DIR, "_temp_processed.jpg")
    cv2.imwrite(temp_proc, processed)

    print("[STAGE 1] Running PaddleOCR ...")
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
    print(f"[STAGE 1] OCR complete in {elapsed:.3f}s  ({len(lines)} lines extracted)")
    return extracted, elapsed


def run_translation_nllb(text, language, translator, tokenizer):
    """
    Translate text from source language to English using NLLB 1.3B via CTranslate2.
    Returns (translated_text, elapsed_seconds).
    """
    if translator is None or tokenizer is None:
        print("[STAGE 2] Skipping translation (source is English)")
        return text, 0.0

    print("[STAGE 2] Translating to English (NLLB 1.3B) ...")
    t0 = time.time()

    src_lang_code      = PIPELINE_LANG_TO_NLLB[language]
    src_token          = NLLB_LANG_MAP[src_lang_code]
    tgt_token          = NLLB_LANG_MAP["en"]
    tokenizer.src_lang = src_token

    if src_lang_code == "zh":
        raw_chunks = re.split(r'(。|？|！)', text)
        chunks     = []
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
    results         = translator.translate_batch(tokenized_batch, target_prefix=target_prefixes)

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


def run_tts(pipertts_module, text):
    """Speak the text through Piper TTS and wait until playback finishes."""
    print("[STAGE 3] Speaking ... (Press any key in terminal to stop)")
    t0 = time.time()
    pipertts_module.speak(text)

    import msvcrt
    stopped = False
    while pipertts_module.is_speaking():
        if msvcrt.kbhit():
            while msvcrt.kbhit():
                msvcrt.getch()
            pipertts_module.stop()
            stopped = True
            break
        time.sleep(0.05)

    pipertts_module.wait_until_done()
    elapsed = time.time() - t0
    if stopped:
        print(f"[STAGE 3] TTS playback stopped by user after {elapsed:.3f}s")
    else:
        print(f"[STAGE 3] TTS playback finished in {elapsed:.3f}s")
    return elapsed


def run_full_pipeline(frame, run_count, language, book_mode,
                      ocr_engine, unwarper, translator, tokenizer,
                      tts_module, pipeline_busy_flag):
    """
    Save the captured frame to disk then run full OCR -> Translate -> TTS.
    Called in a background thread so the live preview stays responsive.
    pipeline_busy_flag is a threading.Event that is cleared when done.
    """
    try:
        print(f"\n{'─'*60}")
        print(f"  RUN #{run_count}  |  Captured from live camera feed")
        print(f"{'─'*60}\n")

        pipeline_t0 = time.time()

        # Save frame to disk
        img_path = save_frame_for_ocr(frame)

        # Stage 1: OCR
        extracted_text, t_ocr = run_ocr(ocr_engine, img_path, book_mode, unwarper)
        if not extracted_text:
            print("[WARN] No text extracted from captured frame. Pipeline skipped.\n")
            return

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
        t_tts = run_tts(tts_module, english_text)

        total_pipeline = time.time() - pipeline_t0

        print(f"\n{'─'*40}")
        print("  TIMING BREAKDOWN (NLLB 1.3B)")
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
        print(f"\n  Ready — press S in the preview window to capture again.\n")

    finally:
        pipeline_busy_flag.clear()  # signal that pipeline is free


# ======================================================================
#  CAMERA STREAM RECEIVER  (based on comms check rx.py)
# ======================================================================
def start_camera_server(port=9999):
    """
    Open a TCP server socket on *port*.
    Waits indefinitely (no timeout) for the Pi to connect.
    No error is thrown if the camera is not yet available — it just waits.
    Returns (server_socket, client_socket) once a client connects.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(5)

    print(f"\n[CAMERA] Listening on TCP port {port} ...")
    print("  Waiting for Smart Glasses (Raspberry Pi) to connect.")
    print("  Will keep waiting indefinitely — no rush!\n")

    # accept() blocks until the Pi connects — this is intentional
    client_socket, addr = server_socket.accept()
    print(f"[CAMERA] Connected to Camera Node at: {addr}")
    return server_socket, client_socket


def receive_frames(client_socket):
    """
    Generator that continuously yields BGR numpy frames from the Pi.
    Stops when the connection is closed.
    """
    data         = b""
    payload_size = struct.calcsize("Q")

    while True:
        # Receive the 8-byte size header
        while len(data) < payload_size:
            packet = client_socket.recv(65536)
            if not packet:
                return
            data += packet

        packed_msg_size = data[:payload_size]
        data            = data[payload_size:]
        msg_size        = struct.unpack("Q", packed_msg_size)[0]

        # Receive the full frame payload
        while len(data) < msg_size:
            data += client_socket.recv(65536)

        frame_data = data[:msg_size]
        data       = data[msg_size:]

        # Decode JPEG payload -> BGR numpy array
        buffer = pickle.loads(frame_data)
        frame  = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        yield frame


# ======================================================================
#  CLI INTERFACE
# ======================================================================
BANNER = r"""
+==============================================================+
|    LIVE CAMERA -> OCR  ->  TRANSLATE  ->  SPEAK  PIPELINE   |
|      Smart Glasses FYDP -- NLLB 1.3B Edition (Pipeline 2)   |
|                                                              |
|  Controls (in preview window):                               |
|      S  =  capture frame and run pipeline                    |
|      Q  =  quit                                              |
+==============================================================+
"""


def ask_language():
    print("\n-- Select source language --")
    print("  1) French")
    print("  2) Chinese")
    print("  3) English (no translation, OCR + TTS only)")
    while True:
        choice = input("\nEnter 1/2/3: ").strip()
        if choice == "1":
            return "french"
        elif choice == "2":
            return "chinese"
        elif choice == "3":
            return "english"
        print("[!] Invalid choice. Enter 1, 2, or 3.")


def ask_book_mode():
    print("\n-- Book Mode (curved page unwarping) --")
    while True:
        choice = input("Enable book mode? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            return True
        elif choice in ("n", "no"):
            return False
        print("[!] Enter y or n.")


# ======================================================================
#  MAIN
# ======================================================================
def main():
    print(BANNER)

    # ---- Step 1: Configuration ----
    language  = ask_language()
    book_mode = ask_book_mode()

    print("\n" + "=" * 60)
    print(f"  Language   : {language.upper()}")
    print(f"  Book Mode  : {'ON' if book_mode else 'OFF'}")
    print(f"  Translator : NLLB-200 Distilled 1.3B (CTranslate2)")
    print("=" * 60)

    # ---- Step 2: Load ALL models once ----
    print("\n[INFO] Loading all models (one-time cost) ...\n")
    total_t0 = time.time()

    ocr_engine            = load_ocr_engine()
    unwarper              = load_unwarper() if book_mode else None
    translator, tokenizer = load_nllb_translator(language)
    tts_module            = load_tts()

    print(f"\n[INFO] All models loaded in {time.time()-total_t0:.2f}s")
    print("=" * 60)

    # ---- Step 3: Start camera server (waits patiently for Pi) ----
    server_socket, client_socket = start_camera_server(port=9999)

    # ---- Step 4: Live preview + capture loop ----
    run_count       = 0
    pipeline_busy   = threading.Event()   # set = busy, clear = free
    pipeline_thread = None

    print("\n[CAMERA] Live feed active.")
    print("  Press  S  in the preview window to capture and run pipeline.")
    print("  Press  Q  to quit.\n")

    try:
        for frame in receive_frames(client_socket):

            # Scale down preview if wider than screen
            h, w = frame.shape[:2]
            max_display_w = 1280
            if w > max_display_w:
                scale   = max_display_w / w
                preview = cv2.resize(frame, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
            else:
                preview = frame.copy()

            # Overlay status text on preview
            is_busy      = pipeline_busy.is_set()
            status_color = (0, 140, 255) if is_busy else (0, 200, 0)
            status_text  = "Pipeline running... please wait" if is_busy else "Press S to capture"
            cv2.putText(preview, status_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        status_color, 2, cv2.LINE_AA)

            cv2.imshow("Live Stream: Smart Glasses Camera", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n[EXIT] Q pressed — shutting down ...")
                break

            elif key == ord('s'):
                if pipeline_busy.is_set():
                    print("[WARN] Pipeline still running — please wait before capturing again.")
                    continue

                run_count     += 1
                captured_frame = frame.copy()  # snapshot NOW before next frame overwrites
                pipeline_busy.set()

                pipeline_thread = threading.Thread(
                    target=run_full_pipeline,
                    args=(captured_frame, run_count, language, book_mode,
                          ocr_engine, unwarper, translator, tokenizer,
                          tts_module, pipeline_busy),
                    daemon=True,
                )
                pipeline_thread.start()

    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"\n[CAMERA] Connection lost: {e}")

    finally:
        if pipeline_thread is not None and pipeline_thread.is_alive():
            print("[INFO] Waiting for current pipeline run to finish ...")
            pipeline_thread.join()

        try:
            client_socket.close()
        except Exception:
            pass
        try:
            server_socket.close()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print(f"\n[EXIT] Session ended. Total captures processed: {run_count}")


if __name__ == "__main__":
    main()
