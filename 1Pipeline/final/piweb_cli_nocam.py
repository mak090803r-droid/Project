"""
piweb_cli_nocam.py  (Final Demo — No Camera Client)
=====================================================
Raspberry Pi (or PC) TCP client that sends static image files to the PC
receiver (pipeline_cli_nocam.py) over port 9999.
No camera required. You type the path of an image file on the client's
CLI terminal and it is sent to the PC pipeline.

Demo Flow:
  1. Connect to PC pipeline server
  2. Prompt for image path
  3. Load image from disk, encode, and send
  4. Repeat until 'q' is typed to quit

Usage:
    python piweb_cli_nocam.py
"""

import os
import sys
import time
import json
import struct
import pickle
import socket
import cv2

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PC_IP        = '192.168.137.1'
PORT         = 9999
RETRY_DELAY  = 3
JPEG_QUALITY = 95

# ══════════════════════════════════════════════════════════════════════════════
#  WIRE PROTOCOL — shared with pipeline_cli_nocam.py
# ══════════════════════════════════════════════════════════════════════════════
MSG_JSON  = 0x01
MSG_FRAME = 0x02


def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!BI", MSG_JSON, len(data)) + data)


def send_frame(sock, jpeg_buf):
    data = pickle.dumps(jpeg_buf)
    sock.sendall(struct.pack("!BQ", MSG_FRAME, len(data)) + data)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def ask_image_path():
    while True:
        path = input("\nEnter image path to send (or 'q' to quit): ").strip().strip('"').strip("'")
        if path.lower() == "q":
            return ""
        if os.path.isfile(path):
            return path
        print(f"[!] File not found: {path}")


def main():
    print("═" * 58)
    print("  piweb_cli_nocam.py — Static Image Sender (No Camera)")
    print("═" * 58)

    # Connect to PC
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            print(f"⏳ Connecting to PC at {PC_IP}:{PORT} …")
            sock.connect((PC_IP, PORT))
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("✅ Connected to PC receiver!")
            break
        except (ConnectionRefusedError, OSError, socket.timeout):
            print(f"❌ PC receiver not ready. Retrying in {RETRY_DELAY}s …")
            sock.close()
            time.sleep(RETRY_DELAY)

    # Send loop
    try:
        while True:
            path = ask_image_path()
            if not path:
                try:
                    send_json(sock, {"cmd": "quit"})
                except Exception:
                    pass
                break

            # Read image file bytes directly to send it "as it is"
            try:
                with open(path, 'rb') as f:
                    raw_bytes = f.read()
                import numpy as np
                buf = np.frombuffer(raw_bytes, dtype=np.uint8)
            except Exception as e:
                print(f"[!] Failed to read image file: {e}")
                continue

            # Send over TCP
            try:
                print(f"📤 Sending image '{os.path.basename(path)}' ({len(buf)/1024:.1f} KB) …")
                send_frame(sock, buf)
                print("✅ Image sent successfully!")
            except (ConnectionError, OSError) as e:
                print(f"[ERROR] Connection lost: {e}")
                break

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        sock.close()
        print("Goodbye.")


if __name__ == "__main__":
    main()
