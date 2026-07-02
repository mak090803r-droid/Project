import cv2
import socket
import struct
import pickle
import numpy as np
import os
import time

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
script_dir = os.path.dirname(os.path.abspath(__file__))
pics_dir   = os.path.abspath(os.path.join(script_dir, "..", "pics"))
os.makedirs(pics_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SERVER SOCKET
# ══════════════════════════════════════════════════════════════
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)q
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(5)
print("📥 WINDOWS AI ENGINE ONLINE: Standing by for Smart Glasses stream...")
print("    Controls:  S = save frame  |  Q = quit")

# Wait for connection
client_socket, addr = server_socket.accept()
print(f"✔️  Connected to Camera Node at: {addr}")

data         = b""
payload_size = struct.calcsize("Q")
frame_count  = 0

# ══════════════════════════════════════════════════════════════
# SHARPENING HELPER — Unsharp Mask (much better than a raw kernel)
# ══════════════════════════════════════════════════════════════
def unsharp_mask(image, sigma=1.0, strength=1.5):
    """
    Apply unsharp-mask sharpening.
    sigma   — Gaussian blur radius (controls what counts as 'detail')
    strength — How much to amplify detail (1.0 = no change, 2.0 = strong)
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

try:
    while True:
        # ── Receive the payload size header ──
        while len(data) < payload_size:
            packet = client_socket.recv(65536)
            if not packet:
                break
            data += packet

        if not data:
            break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # ── Receive the full frame payload ──
        while len(data) < msg_size:
            data += client_socket.recv(65536)

        frame_data = data[:msg_size]
        data       = data[msg_size:]

        # ── Decode ──
        buffer = pickle.loads(frame_data)
        frame  = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # No color conversion needed — Pi sends BGR-encoded JPEG,
        # imdecode returns BGR. Colors are correct out of the box.

        frame_count += 1

        # =========================================================
        # 🔥 YOUR AI PIPELINE INTEGRATES HERE
        # =========================================================
        # 'frame' is a full 1920×1080 BGR numpy array.
        # Feed it directly to your model:
        #   ocr_result = ocr_engine.ocr(frame, cls=True)
        # =========================================================

        # ── Display a preview (scale down only if larger than screen) ──
        h, w = frame.shape[:2]
        max_display_w = 1280
        if w > max_display_w:
            scale    = max_display_w / w
            preview  = cv2.resize(frame, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
        else:
            preview = frame

        cv2.imshow("Live Stream: Raspberry Pi Smart Glasses", preview)

        # ── Keyboard handling ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            # Save the FULL-RESOLUTION frame (not the preview)
            # Apply unsharp-mask sharpening for maximum OCR readability
            sharpened = unsharp_mask(frame, sigma=1.0, strength=1.5)

            # Save as lossless PNG so OCR gets every pixel
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename  = f"capture_{timestamp}.png"
            filepath  = os.path.join(pics_dir, filename)
            cv2.imwrite(filepath, sharpened)
            print(f"💾 Saved sharpened frame ({frame.shape[1]}×{frame.shape[0]}) → {filepath}")

finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print(f"Session ended. {frame_count} frames received.")