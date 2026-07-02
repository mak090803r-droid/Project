import os
import cv2
import numpy as np
import time
from paddleocr import PaddleOCR

# ══════════════════════════════════════════════
# OPTIMIZED TUNING KNOBS (Single-Channel Focus)
# ══════════════════════════════════════════════
CONFIG = {
    "upscale_factor":   1.5,   
    "clahe_clip":       2,   
    "clahe_grid":       8,     
    "gamma":            0.7,   # 1.2 with inv_gamma brightens midtones safely
    "sharpen":          False,  
    "sharpen_strength": 1.0,   
}

# Pre-compute performance configurations ONCE at startup to save mid-frame CPU cycles
INV_GAMMA = 1.0 / CONFIG["gamma"]
GAMMA_LUT = np.array([
    ((i / 255.0) ** INV_GAMMA) * 255
    for i in range(256)
]).astype("uint8")

CLAHE_ENGINE = cv2.createCLAHE(
    clipLimit=CONFIG["clahe_clip"],
    tileGridSize=(CONFIG["clahe_grid"], CONFIG["clahe_grid"])
)

# Pre-build sharpening matrix
S = CONFIG["sharpen_strength"]
SHARPEN_KERNEL = np.array([
    [-S,    -S,   -S],
    [-S, 1+8*S,   -S],
    [-S,    -S,   -S]
])

def preprocess_optimized(img_path, cfg=CONFIG, save_debug=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read: {img_path}")
        return None

    # Step 1: Immediately drop to Grayscale (1 Channel instead of 3)
    # Everything below here now runs 3x faster and consumes 1/3 of the RAM
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Denoise on the small grayscale layout
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3: Streamlined CLAHE (No LAB splitting or merging required!)
    contrast = CLAHE_ENGINE.apply(gray)

    # Step 4: Streamlined Gamma LUT mapping on a single channel
    enhanced = cv2.LUT(contrast, GAMMA_LUT)

    # Step 5: Sharpen edge transitions
    if cfg["sharpen"]:
        enhanced = cv2.filter2D(enhanced, -1, SHARPEN_KERNEL)

    # Step 6: Upscale LAST (INTER_LINEAR is faster and smoother for deep learning)
    h, w = enhanced.shape[:2]
    final_img = cv2.resize(
        enhanced,
        (int(w * cfg["upscale_factor"]), int(h * cfg["upscale_factor"])),
        interpolation=cv2.INTER_LINEAR
    )

    if True:
        cv2.imwrite("debug_preprocessed.jpg", final_img)
        print(f"[DEBUG] {w}x{h} → {final_img.shape[1]}x{final_img.shape[0]} (Single Channel)")

    return final_img

# ══════════════════════════════════════════════
# MAIN RUNTIME
# ══════════════════════════════════════════════
if __name__ == '__main__':
    start = time.time()

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="paddle",
        enable_mkldnn=False
    )

    img_path = "testimg2.jpg"

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
    else:
        print("[INFO] Running optimized preprocessing...")
        t1 = time.time()
        processed = preprocess_optimized(img_path, save_debug=True)
        t2 = time.time()

        temp_path = "temp_processed.jpg"
        cv2.imwrite(temp_path, processed)

        print("[INFO] Running OCR...")
        t3 = time.time()
        result = ocr.predict(temp_path)
        t4 = time.time()

        paragraph_lines = []
        for res in result:
            res.save_to_img("output")
            res.save_to_json("output")
            if 'rec_texts' in res:
                for text in res['rec_texts']:
                    if text.strip():
                        paragraph_lines.append(text.strip())

        full_paragraph = " ".join(paragraph_lines)

        print("\n" + "="*60)
        print("              EXTRACTED PARAGRAPH")
        print("="*60)
        print(full_paragraph)
        print("="*60)
        
        print(f"\n[TIMING BREAKDOWN]")
        print(f"  New Preprocessing : {t2-t1:.3f}s  <-- Notice the drop!")
        print(f"  OCR inference     : {t4-t3:.3f}s")
        print(f"  Total Execution   : {time.time()-start:.3f}s")

        if os.path.exists(temp_path):
            os.remove(temp_path)