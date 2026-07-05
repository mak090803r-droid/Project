"""
pipeline_nllb1.3b.py
====================
Integrated OCR → Translation → TTS pipeline.
Uses NLLB-200 Distilled 1.3B (via CTranslate2) for translation.

Usage:
    python pipeline_nllb1.3b.py

On launch you pick:
    1. Source language  (French / Chinese / English-only)
    2. Book mode        (curved-page unwarping via UVDoc)
    3. Image path       (the photo to process)

After each run the pipeline loops — just supply the next image path.
All heavy models (OCR, Translator, TTS) are loaded ONCE at startup.
"""

import os
import sys
import re
import time
import cv2
import numpy as np
import torch
import ctranslate2
import transformers

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING  (from paddleocr for curved pages.py)
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


def preprocess_image(img_path: str) -> np.ndarray | None:
    """Grayscale preprocessing pipeline: denoise → CLAHE → gamma → sharpen → upscale."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
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


# ══════════════════════════════════════════════════════════════════════════════
#  NLLB 1.3B  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# Hardware acceleration detection
if torch.cuda.is_available():
    NLLB_DEVICE = "cuda"
    NLLB_COMPUTE_TYPE = "float16"
    os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
    print("[INFO] Hardware Target: NVIDIA GPU (CUDA Optimization Active)")
else:
    NLLB_DEVICE = "cpu"
    NLLB_COMPUTE_TYPE = "int8"
    print("[INFO] Hardware Target: CPU ONLY (Fallback Int8 Execution Active)")

NLLB_MODEL_DIR = "nllb-200-1.3B-ct2"
NLLB_BASE_MODEL = "facebook/nllb-200-distilled-1.3B"

# FLORES-200 target language vector designators
NLLB_LANG_MAP = {
    "zh": "zho_Hans",  # Chinese (Simplified)
    "fr": "fra_Latn",  # French
    "en": "eng_Latn"   # English
}

# Language code mapping from pipeline language names to NLLB codes
PIPELINE_LANG_TO_NLLB = {
    "french":  "fr",
    "chinese": "zh",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_ocr_engine():
    """Load PaddleOCR engine (done once). Uses ONNX Runtime backend for GPU without pybind11 conflicts."""
    from paddleocr import PaddleOCR
    ocr_device = "gpu" if NLLB_DEVICE == "cuda" else "cpu"
    print(f"[LOAD] Initializing PaddleOCR engine (engine=onnxruntime, device={ocr_device}) …")
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
    """Load UVDoc text-image unwarper for book mode."""
    from paddleocr import TextImageUnwarping
    print("[LOAD] Initializing UVDoc unwarper (book mode) …")
    t = time.time()
    unwarper = TextImageUnwarping(model_name="UVDoc", engine="paddle")
    print(f"[LOAD] UVDoc ready in {time.time()-t:.2f}s")
    return unwarper


def load_nllb_translator(language: str):
    """
    Load NLLB-200 1.3B Distilled model via CTranslate2.
    Returns (translator, tokenizer) or (None, None) for English.
    """
    if language == "english":
        return None, None  # no translation needed

    print(f"[LOAD] Loading NLLB-200 1.3B Distilled translator …")
    t = time.time()

    # Auto-convert model to CTranslate2 format if not already done
    if not os.path.exists(NLLB_MODEL_DIR):
        print(f"[INFO] Compiling raw {NLLB_BASE_MODEL} into optimized CTranslate2 binary …")
        start_conv = time.time()
        conversion_cmd = (
            f"ct2-transformers-converter --model {NLLB_BASE_MODEL} "
            f"--output_dir {NLLB_MODEL_DIR} --force --quantization {NLLB_COMPUTE_TYPE}"
        )
        exit_code = os.system(conversion_cmd)
        if exit_code != 0:
            raise RuntimeError("[ERROR] Model compilation failed. Verify system dependencies.")
        print(f"[INFO] Serialization complete! Saved to '{NLLB_MODEL_DIR}' ({time.time()-start_conv:.2f}s)")

    # Initialize CTranslate2 translator
    translator = ctranslate2.Translator(NLLB_MODEL_DIR, device=NLLB_DEVICE, compute_type=NLLB_COMPUTE_TYPE)

    # Load tokenizer from the base HuggingFace model
    tokenizer = transformers.AutoTokenizer.from_pretrained(NLLB_BASE_MODEL)

    print(f"[LOAD] NLLB 1.3B translator ready in {time.time()-t:.2f}s")
    return translator, tokenizer


def load_tts():
    """Pre-load Piper TTS voice model."""
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

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
def run_ocr(ocr_engine, img_path: str, book_mode: bool, unwarper) -> tuple[str, float]:
    """Run OCR on an image. Returns (extracted_text, elapsed_seconds)."""
    t0 = time.time()

    # --- Optional: unwarp curved book pages ---
    if book_mode and unwarper is not None:
        print("[STAGE 1] Unwarping curved page …")
        unwarp_result = unwarper.predict(img_path, batch_size=1)
        for res in unwarp_result:
            unwarped_img = res["doctr_img"]
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_unwarped.jpg")
        cv2.imwrite(temp_path, unwarped_img)
        img_path = temp_path

    # --- Preprocess ---
    print("[STAGE 1] Preprocessing image …")
    processed = preprocess_image(img_path)
    if processed is None:
        return "", time.time() - t0
    temp_proc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_processed.jpg")
    cv2.imwrite(temp_proc, processed)

    # --- OCR ---
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
    for f in [temp_proc, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_unwarped.jpg")]:
        if os.path.exists(f):
            os.remove(f)

    elapsed = time.time() - t0
    print(f"[STAGE 1] OCR complete in {elapsed:.3f}s  ({len(lines)} lines extracted)")
    return extracted, elapsed


def run_translation_nllb(text: str, language: str, translator, tokenizer) -> tuple[str, float]:
    """
    Translate text from source language → English using NLLB 1.3B via CTranslate2.
    Splits into sentences for batch translation. Returns (translated_text, elapsed_seconds).
    """
    if translator is None or tokenizer is None:
        print("[STAGE 2] Skipping translation (source is English)")
        return text, 0.0

    print("[STAGE 2] Translating to English (NLLB 1.3B) …")
    t0 = time.time()

    src_lang_code = PIPELINE_LANG_TO_NLLB[language]
    src_token = NLLB_LANG_MAP[src_lang_code]
    tgt_token = NLLB_LANG_MAP["en"]

    # Update tokenizer source language
    tokenizer.src_lang = src_token

    # --- Sentence Segmentation ---
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

    # --- Batch Tokenization ---
    tokenized_batch = []
    for chunk in chunks:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk))
        tokenized_batch.append(tokens)

    if not tokenized_batch:
        return text, time.time() - t0

    # --- Parallel Inference ---
    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    results = translator.translate_batch(tokenized_batch, target_prefix=target_prefixes) #target_prefix=target_prefixes, beam_size=1) can do this for increase translation speed

    # --- Decode & Reassemble ---
    translated_sentences = []
    for res in results:
        output_tokens = res.hypotheses[0]
        raw_decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        clean_sentence = raw_decoded.replace(tgt_token, "").strip()
        translated_sentences.append(clean_sentence)

    result = " ".join(translated_sentences)

    elapsed = time.time() - t0
    print(f"[STAGE 2] Translation complete in {elapsed:.3f}s")
    return result, elapsed


def run_tts(pipertts_module, text: str) -> float:
    """Speak the text through Piper TTS and wait until playback finishes.
    Returns elapsed_seconds."""
    print("[STAGE 3] Speaking … (Press any key to stop and request a new image)")
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


# ══════════════════════════════════════════════════════════════════════════════
#  CLI  INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║           OCR  →  TRANSLATE  →  SPEAK  PIPELINE             ║
║          Smart Glasses FYDP — NLLB 1.3B Edition             ║
╚══════════════════════════════════════════════════════════════╝
"""


def ask_language() -> str:
    print("\n── Select source language ──")
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


def ask_book_mode() -> bool:
    print("\n── Book Mode (curved page unwarping) ──")
    while True:
        choice = input("Enable book mode? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            return True
        elif choice in ("n", "no"):
            return False
        print("[!] Enter y or n.")


def ask_image_path() -> str:
    while True:
        path = input("\nImage path (or 'q' to quit): ").strip().strip('"').strip("'")
        if path.lower() == "q":
            return ""
        if os.path.isfile(path):
            return path
        print(f"[!] File not found: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
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
    print("\n[INFO] Loading all models (one-time cost) …\n")
    total_t0 = time.time()

    ocr_engine = load_ocr_engine()
    unwarper   = load_unwarper() if book_mode else None
    translator, tokenizer = load_nllb_translator(language)
    tts_module = load_tts()

    print(f"\n[INFO] All models loaded in {time.time()-total_t0:.2f}s")
    print("=" * 60)

    # ---- Step 3: Image processing loop ----
    run_count = 0
    while True:
        img_path = ask_image_path()
        if not img_path:
            print("\n[EXIT] Goodbye!")
            break

        run_count += 1
        print(f"\n{'─'*60}")
        print(f"  RUN #{run_count}  |  {os.path.basename(img_path)}")
        print(f"{'─'*60}\n")

        pipeline_t0 = time.time()

        # Stage 1: OCR
        extracted_text, t_ocr = run_ocr(ocr_engine, img_path, book_mode, unwarper)
        if not extracted_text:
            print("[WARN] No text extracted. Skipping translation & TTS.\n")
            continue

        print(f"\n{'='*60}")
        print("  EXTRACTED TEXT")
        print(f"{'='*60}")
        print(extracted_text)
        print(f"{'='*60}\n")

        # Stage 2: Translation (NLLB 1.3B)
        english_text, t_translate = run_translation_nllb(extracted_text, language, translator, tokenizer)

        if language != "english":
            print(f"\n{'='*60}")
            print("  TRANSLATED TEXT (English) — NLLB 1.3B")
            print(f"{'='*60}")
            print(english_text)
            print(f"{'='*60}\n")

        # Stage 3: TTS
        t_tts = run_tts(tts_module, english_text)

        total_pipeline = time.time() - pipeline_t0

        # ── Timing Summary ─────────────────────────────────────────
        print(f"\n{'─'*40}")
        print("  ⏱  TIMING BREAKDOWN (NLLB 1.3B)")
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
        print(f"\n  Ready for next image!\n")


if __name__ == "__main__":
    main()
