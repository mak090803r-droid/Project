import cv2
import socket
import pickle
import struct
import time
import numpy as np
from picamera2 import Picamera2

# ══════════════════════════════════════════════════════════════
# NETWORK CONFIG
# ══════════════════════════════════════════════════════════════
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PC_IP = '192.168.137.1'
PORT  = 9999

print(f"Connecting to Windows PC at {PC_IP}:{PORT}...")
client_socket.connect((PC_IP, PORT))
print("Connection established!")

# ══════════════════════════════════════════════════════════════
# CAMERA CONFIG — Optimized for OCR-quality text capture
# ══════════════════════════════════════════════════════════════
print("Initializing Pi Camera...")
picam2 = Picamera2()

# Use 1920x1080 — high resolution without the extreme bandwidth
# of full 5MP, and avoids the heavy center-crop of lower res modes.
# The ISP still uses the full sensor area and downscales, so you
# get the full field of view.
config = picam2.create_video_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},
    controls={
        "FrameRate":          15,       # 15fps is plenty for OCR work
        "Sharpness":          3.0,      # Moderate-high sharpness for crisp text edges
        "Contrast":           1.2,      # Slight contrast boost for text/background separation
        "Brightness":         0.0,      # Neutral brightness — let AE handle it
        "Saturation":         1.0,      # Normal color saturation
        "NoiseReductionMode": 2,        # High quality noise reduction
        "AwbEnable":          True,     # Auto white balance ON
        "AeEnable":           True,     # Auto exposure ON
    }
)
picam2.configure(config)
picam2.start()

# Let auto-exposure and white balance stabilize
print("Waiting for camera sensors to stabilize...")
time.sleep(3)
print("Camera ready! Streaming...")

try:
    while True:
        # Capture frame — picamera2 RGB888 gives us RGB-ordered pixels
        frame_rgb = picam2.capture_array()

        # CRITICAL: Convert RGB (picamera2) -> BGR (OpenCV) using numpy
        # This is a direct memory channel swap — guaranteed correct.
        # cv2.imencode then correctly encodes this BGR frame into JPEG,
        # and cv2.imdecode on the receiver correctly gets BGR back.
        frame_bgr = frame_rgb[:, :, ::-1].copy()

        # Encode as near-lossless JPEG (quality 98)
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if not ret:
            continue

        data = pickle.dumps(buffer)

        # Pack size header + payload and send
        message = struct.pack("Q", len(data)) + data
        client_socket.sendall(message)

except Exception as e:
    print(f"Stream interrupted: {e}")

finally:
    picam2.stop()
    client_socket.close()
    print("Cleaned up. Goodbye.")