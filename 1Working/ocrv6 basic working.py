#Install pip install paddleocr
#then cpu verison do  install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ 


import os
from paddleocr import PaddleOCR
import time

start=time.time()
# 1. Initialize the official High-Level PP-OCRv5 Pipeline
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
   text_detection_model_name="PP-OCRv6_medium_det",  # UPDATED: Switched to v6 Mobile Detection
    text_recognition_model_name="PP-OCRv6_medium_rec",
    use_doc_unwarping=False,
    use_textline_orientation=False,
    engine="paddle",
    enable_mkldnn=False  # <--- THIS IS THE CRITICAL BUG FIX
)

# 2. Assign target image file pathway
img_path = "testimg2.jpg"

if not os.path.exists(img_path):
    print(f"\n[ERROR] Could not find file at: {img_path}")
    print("-> Please drag and drop 'testimg2.jpg' into the Files menu tab on the left panel of Colab.")
else:
    print(f"\n[INFO] Starting PP-OCRv5 inference engine on target image...")
    
    # 3. Run prediction pipeline
    result = ocr.predict(img_path)
    
    # --- PARAGRAPH STORAGE LIST ---
    paragraph_lines = []
    
    # 4. Loop through the output container object, save data, and extract text strings
    for res in result:
        res.save_to_img("output")  # Saves visual verification image to workspace
        res.save_to_json("output")  # Saves perfectly structured JSON file
        
        # FIXED: Look directly inside the res dictionary keys for 'rec_texts'
        if 'rec_texts' in res:
            for text in res['rec_texts']:
                paragraph_lines.append(text.strip())

    # 5. Join all extracted text blocks together into a single paragraph string
    full_paragraph = " ".join(paragraph_lines)

    print("\n" + "="*60)
    print("                 EXTRACTED PARAGRAPH                          ")
    print("="*60)
    print(full_paragraph)
    print("="*60)
    print("[SUCCESS] Check the 'output' directory folder on your left panel to see files!")

end = time.time()
print("Execution time:", end - start, "seconds")