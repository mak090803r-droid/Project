import importlib.util
import os

import cv2


BOX3D_PATH = os.path.join(os.path.dirname(__file__), "pipeline_cli_box3d.py")
spec = importlib.util.spec_from_file_location("pipeline_cli_box3d", BOX3D_PATH)
box3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box3d)

captured_dir = os.path.join(box3d.PROJECT_DIR, "pics", "captured")
image_names = [
    "capture_cli_003_20260715_153129.jpg",
]
output_dir = os.path.join(os.path.dirname(__file__), "paragraph_test_outputs")
os.makedirs(output_dir, exist_ok=True)

ocr_engine = box3d.load_ocr_engine("english")
unwarper = box3d.load_unwarper()

for image_name in image_names:
    image_path = os.path.join(captured_dir, image_name)
    extracted, paragraphs, ocr_image, elapsed = box3d.run_ocr(
        ocr_engine, image_path, True, unwarper)
    print(f"\n[TEST_IMAGE] {image_name}")
    print(
        f"[TEST_RESULT] paragraphs={len(paragraphs)} "
        f"characters={len(extracted)} elapsed={elapsed:.3f}s")
    for paragraph in paragraphs:
        preview = paragraph["source_text"].replace("\n", " ")[:180]
        print(
            f"  P{paragraph['number']} lines={len(paragraph['lines'])} "
            f"confidence={paragraph['confidence']:.3f} "
            f"bbox={paragraph['bbox']} :: {preview}")

    overlay = box3d.draw_paragraph_overlay(ocr_image, paragraphs)
    output_path = os.path.join(output_dir, f"paragraphs_{image_name}")
    cv2.imwrite(output_path, overlay)
    print(f"[TEST_OVERLAY] {output_path}")
