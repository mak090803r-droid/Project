"""
pipeline.py
===========
Integrated OCR → Translation → TTS pipeline.

Usage:
    python pipeline.py

On launch you pick:
    1. Source language  (French / Chinese / English-only)
    2. Book mode        (curved-page unwarping via UVDoc)
    3. Image path       (the photo to process)

After each run the pipeline loops — just supply the next image path.
All heavy models (OCR, Translator, TTS) are loaded ONCE at startup.
"""

import os
import sys
import time
import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING  (from paddleocr for curved pages.py)
# ══════════════════════════════════════════════════════════════════════════════
PREPROCESS_CONFIG = {
    "upscale_factor":   1.5,
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
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_ocr_engine():
    """Load PaddleOCR engine (done once)."""
    from paddleocr import PaddleOCR
    print("[LOAD] Initializing PaddleOCR engine …")
    t = time.time()
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="paddle",
        enable_mkldnn=False,
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


def load_translator(language: str):
    """Load MarianMT model for the chosen source language → English."""
    if language == "english":
        return None, None  # no translation needed

    from transformers import MarianMTModel, MarianTokenizer

    model_map = {
        "french":  "Helsinki-NLP/opus-mt-fr-en",
        "chinese": "Helsinki-NLP/opus-mt-zh-en",
    }
    model_name = model_map[language]
    print(f"[LOAD] Loading MarianMT model: {model_name} …")
    t = time.time()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print(f"[LOAD] Translator ready in {time.time()-t:.2f}s")
    return tokenizer, model


def load_tts():
    """Pre-load Piper TTS voice model."""
    # pipertts.py sits in the same directory — import its public API
    # We need to add the pipeline directory to sys.path if not already there
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


def run_translation(text: str, tokenizer, model) -> tuple[str, float]:
    """Translate text from source language → English using MarianMT.
    Returns (translated_text, elapsed_seconds). elapsed=0 when skipped."""
    if tokenizer is None or model is None:
        print("[STAGE 2] Skipping translation (source is English)")
        return text, 0.0

    print("[STAGE 2] Translating to English …")
    t0 = time.time()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"[STAGE 2] Translation complete in {elapsed:.3f}s")
    return result, elapsed


def run_tts(pipertts_module, text: str) -> float:
    """Speak the text through Piper TTS and wait until playback finishes.
    Returns elapsed_seconds."""
    print("[STAGE 3] Speaking …")
    t0 = time.time()
    pipertts_module.speak(text)
    pipertts_module.wait_until_done()
    elapsed = time.time() - t0
    print(f"[STAGE 3] TTS playback finished in {elapsed:.3f}s")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  CLI  INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║           OCR  →  TRANSLATE  →  SPEAK  PIPELINE             ║
║                Smart Glasses FYDP — v1.0                     ║
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
    print(f"  Language  : {language.upper()}")
    print(f"  Book Mode : {'ON' if book_mode else 'OFF'}")
    print("=" * 60)

    # ---- Step 2: Load ALL models once ----
    print("\n[INFO] Loading all models (one-time cost) …\n")
    total_t0 = time.time()

    ocr_engine = load_ocr_engine()
    unwarper   = load_unwarper() if book_mode else None
    tokenizer, translator = load_translator(language)
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

        # Stage 2: Translation
        english_text, t_translate = run_translation(extracted_text, tokenizer, translator)

        if language != "english":
            print(f"\n{'='*60}")
            print("  TRANSLATED TEXT (English)")
            print(f"{'='*60}")
            print(english_text)
            print(f"{'='*60}\n")

        # Stage 3: TTS
        t_tts = run_tts(tts_module, english_text)

        total_pipeline = time.time() - pipeline_t0

        # ── Timing Summary ─────────────────────────────────────────
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
        print(f"\n  Ready for next image!\n")


if __name__ == "__main__":
    main()
