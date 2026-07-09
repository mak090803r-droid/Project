"""
testformat.py — PNG vs JPEG OCR Speed Comparison
==================================================
Give it ONE PNG image.
It OCRs the PNG directly, then converts it to JPG internally
and OCRs that, then prints timing + text for both.

Usage:
    python testformat.py "C:\\full\\path\\to\\image.png"
"""

import os
import sys
import time
import cv2
import torch

# ── Detect hardware ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    NLLB_DEVICE = "cuda"
    print("[INFO] Hardware: NVIDIA GPU (CUDA)")
else:
    NLLB_DEVICE = "cpu"
    print("[INFO] Hardware: CPU only")

# ── Temp file — always next to this script, never mixed with input path ────────
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_JPG     = os.path.join(PIPELINE_DIR, "_test_converted.jpg")


# ── PaddleOCR loader — same pattern as pipeline_cli.py ────────────────────────
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
    print(f"[LOAD] PaddleOCR ready in {time.time()-t:.2f}s\n")
    return ocr


# ── OCR helper ─────────────────────────────────────────────────────────────────
def run_ocr(ocr_engine, img_path, label):
    print(f"  → OCR on {label} …")
    t0 = time.time()
    result = ocr_engine.predict(img_path)
    elapsed = time.time() - t0

    lines = []
    for res in result:
        if "rec_texts" in res:
            for text in res["rec_texts"]:
                if text.strip():
                    lines.append(text.strip())

    return " ".join(lines), elapsed


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print('Usage: python testformat.py "C:\\full\\path\\to\\image.png"')
        sys.exit(1)

    # Strip quotes the user may have included
    png_path = sys.argv[1].strip('"').strip("'")

    if not os.path.isfile(png_path):
        print(f"[ERROR] File not found: {png_path}")
        sys.exit(1)

    size_png_kb = os.path.getsize(png_path) / 1024

    print("=" * 60)
    print("  PNG vs JPEG — OCR Speed & Quality Test")
    print("=" * 60)
    print(f"  Input PNG : {png_path}")
    print(f"  PNG size  : {size_png_kb:.1f} KB\n")

    ocr_engine = load_ocr_engine()

    # ── Step 1: OCR on original PNG ───────────────────────────────────────────
    print("─" * 60)
    print("  [1] OCR on original PNG")
    print("─" * 60)
    text_png, t_png = run_ocr(ocr_engine, png_path, "PNG")
    print(f"  Time: {t_png:.3f}s")
    print(f"  Text: {text_png[:400] if text_png else '[NO TEXT DETECTED]'}\n")

    # ── Step 2: Convert PNG → JPEG internally ────────────────────────────────
    print("─" * 60)
    print("  [2] Converting PNG → JPEG (quality=95) …")
    img = cv2.imread(png_path)
    if img is None:
        print(f"[ERROR] cv2 could not open: {png_path}")
        sys.exit(1)
    t_conv = time.time()
    cv2.imwrite(TEMP_JPG, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elapsed_conv = time.time() - t_conv
    size_jpg_kb = os.path.getsize(TEMP_JPG) / 1024
    print(f"  JPEG size : {size_jpg_kb:.1f} KB  (converted in {elapsed_conv:.3f}s)\n")

    # ── Step 3: OCR on converted JPEG ────────────────────────────────────────
    print("  [3] OCR on converted JPEG")
    print("─" * 60)
    text_jpg, t_jpg = run_ocr(ocr_engine, TEMP_JPG, "JPEG")
    print(f"  Time: {t_jpg:.3f}s")
    print(f"  Text: {text_jpg[:400] if text_jpg else '[NO TEXT DETECTED]'}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  📄 PNG  — {size_png_kb:.1f} KB — OCR: {t_png:.3f}s")
    print(f"  🖼  JPEG — {size_jpg_kb:.1f} KB — OCR: {t_jpg:.3f}s  (+ {elapsed_conv:.3f}s to convert)")
    diff = t_jpg - t_png
    if diff < 0:
        print(f"\n  ⚡ JPEG was faster by {abs(diff):.3f}s")
    elif diff > 0:
        print(f"\n  ⚡ PNG  was faster by {abs(diff):.3f}s")
    else:
        print("\n  ⚡ Both took the same time")
    print(f"  💾 JPEG is {size_png_kb - size_jpg_kb:.1f} KB smaller than PNG")
    print("=" * 60)

    # Cleanup temp file
    if os.path.exists(TEMP_JPG):
        os.remove(TEMP_JPG)


if __name__ == "__main__":
    main()
