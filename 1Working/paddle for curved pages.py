import os
import cv2
import numpy as np
import time
from paddleocr import PaddleOCR, TextImageUnwarping

# ══════════════════════════════════════════════
# OPTIMIZED TUNING KNOBS (Single-Channel Focus)
# ══════════════════════════════════════════════
CONFIG = {
    "book_mode":        True, # Toggle True for curved pages, False for normal flat documents
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if save_debug:
        cv2.imwrite("debug_preprocessed.jpg", gray)
        print(f"[DEBUG] Saved grayscale image of shape {gray.shape[1]}x{gray.shape[0]} to debug_preprocessed.jpg")

    return gray

# ══════════════════════════════════════════════
# MAIN RUNTIME
# ══════════════════════════════════════════════
if __name__ == '__main__':
    start = time.time()

    ocr = PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
        device="gpu",
    )
    unwarper = None
    if CONFIG["book_mode"]:
        print("[INFO] Book Mode Active: Loading UVDoc Engine...")
        unwarper = TextImageUnwarping(model_name="UVDoc", engine="paddle")

    # Automatically resolve the correct path to the captured folder relative to the script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(SCRIPT_DIR, "..", "pics", "captured", "capture_cli_001_20260716_211940.jpg").replace("\\", "/")

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
    else:
        print("[INFO] Running optimized preprocessing...")
        t1 = time.time()
        
        if CONFIG["book_mode"]:
            unwarp_result = unwarper.predict(img_path, batch_size=1)
            
            # Extract the corrected frame matrix using the direct top-level key
            for res in unwarp_result:
                unwarped_img = res['doctr_img']
            
            # Save the raw unwrapped image for the user
            raw_unwrapped_path = "unwrapped_page.jpg"
            cv2.imwrite(raw_unwrapped_path, unwarped_img)
            print(f"[INFO] Raw unwrapped image saved to: {raw_unwrapped_path}")
            
            temp_path = "temp_unwarped.jpg"
            cv2.imwrite(temp_path, unwarped_img)
            
            processed = preprocess_optimized(temp_path, save_debug=True)
            ocr_input_path = "temp_processed.jpg"
            cv2.imwrite(ocr_input_path, processed)
            temp_path = ocr_input_path # Set temp_path to ocr_input_path so ocr.predict(temp_path) runs on the processed image
        else:
            processed = preprocess_optimized(img_path, save_debug=True)
            temp_path = "temp_processed.jpg"
            cv2.imwrite(temp_path, processed)
            
        t2 = time.time()

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