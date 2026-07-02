# stream_receiver.py
# Install: pip install opencv-python requests numpy
# Run: python stream_receiver.py

import cv2
import time
import requests
import numpy as np

# ══════════════════════════════════════════
# CONFIG — paste sender IP here
# ══════════════════════════════════════════
SENDER_IP   = "192.168.137.210" 
  # <-- change to your laptop/Pi IP
PORT        = 8080
STREAM_URL  = "http://192.168.137.210:8080/stream"
CAPTURE_URL = "http://192.168.137.210:8080/capture"
STATUS_URL  = "http://192.168.137.210:8080/status"


# ══════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════
def check_connection():
    """Verify sender is reachable before doing anything"""
    print(f"[INFO] Connecting to {SENDER_IP}:{PORT}...")
    first_attempt = True
    while True:
        try:
            r = requests.get(STATUS_URL, timeout=4)
            if r.status_code == 200:
                info = r.json()
                print(f"[OK] Connected!")
                print(f"[OK] Camera      : {info['camera']}")
                print(f"[OK] Stream res  : {info['stream_res']}")
                print(f"[OK] Capture res : {info['capture_res']}")
                print(f"[OK] JPEG quality: {info['jpeg_quality']}")
                return True
        except requests.exceptions.ConnectionError:
            if first_attempt:
                print(f"[ERROR] Cannot reach sender at {SENDER_IP}:{PORT}")
                print(f"        -> Make sure stream_sender.py is running")
                print(f"        -> Make sure both on same network")
                print(f"        -> Try: http://{SENDER_IP}:{PORT}/status in browser")
                print(f"[INFO] Waiting for connection (Press Ctrl+C to cancel)...")
                first_attempt = False
            else:
                print(f"[INFO] Still trying to connect to {SENDER_IP}:{PORT}...")
        except KeyboardInterrupt:
            print("\n[INFO] Connection cancelled by user. Exiting.")
            return False
        except Exception as e:
            print(f"[ERROR] {e}")
        
        time.sleep(2)



# ══════════════════════════════════════════
# FRAME FUNCTIONS
# ══════════════════════════════════════════
def get_single_frame():
    """
    Grab one high quality frame from sender
    Use this for OCR
    """
    try:
        t1       = time.time()
        response = requests.get(CAPTURE_URL, timeout=5)
        t2       = time.time()

        img_arr  = np.frombuffer(response.content, dtype=np.uint8)
        frame    = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            transfer_ms = (t2 - t1) * 1000
            print(f"[OK] Frame received | "
                  f"Size: {frame.shape[1]}x{frame.shape[0]} | "
                  f"Transfer: {transfer_ms:.0f}ms")
        return frame

    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out — is sender running?")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def get_best_frame(n=3):
    """
    Grab n frames and return the sharpest one
    Eliminates motion blur from camera movement
    Best to use for OCR trigger
    """
    def sharpness_score(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    print(f"[INFO] Capturing {n} frames, picking sharpest...")
    frames = []

    for i in range(n):
        frame = get_single_frame()
        if frame is not None:
            score = sharpness_score(frame)
            frames.append((score, frame))
            print(f"  Frame {i+1}: sharpness {score:.1f}")

    if not frames:
        print("[ERROR] No frames received")
        return None

    frames.sort(key=lambda x: x[0], reverse=True)
    best_score, best_frame = frames[0]
    print(f"[INFO] Best sharpness: {best_score:.1f}")
    return best_frame

# ══════════════════════════════════════════
# LIVE VIEW
# ══════════════════════════════════════════
def live_view():
    """
    Watch live stream from sender
    Use this to position camera on glasses
    Press Q to quit
    Press S to save a frame
    """
    print(f"\n[INFO] Opening live stream...")
    print(f"[INFO] Press Q to quit")
    print(f"[INFO] Press S to save current frame")

    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("[ERROR] Cannot open stream")
        print(f"        Try opening in browser: {STREAM_URL}")
        return

    frame_count = 0
    fps_timer   = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Stream lost — trying to reconnect...")
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_timer   = time.time()

        # Overlay info on frame
        overlay = frame.copy()
        cv2.putText(overlay,
                    f"FPS: {fps_display:.1f}  |  {SENDER_IP}  |  Q=quit  S=save",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

        cv2.imshow("Live Stream — Camera Feed", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"saved_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[OK] Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Live view closed")

# ══════════════════════════════════════════
# MAIN MENU
# ══════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Stream Receiver — Windows PC")
    print(f"  Sender: {SENDER_IP}:{PORT}")
    print(f"{'='*50}\n")

    if not check_connection():
        exit()

    while True:
        print(f"\n{'─'*40}")
        print("  1. Live view  (position camera)")
        print("  2. Single frame capture")
        print("  3. Best frame (sharpest of 3)")
        print("  4. Best frame (sharpest of 5)")
        print("  0. Exit")
        print(f"{'─'*40}")
        choice = input("  Choice: ").strip()

        if choice == "1":
            live_view()

        elif choice == "2":
            frame = get_single_frame()
            if frame is not None:
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] Saved: {filename}")
                cv2.imshow("Captured Frame", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif choice == "3":
            frame = get_best_frame(n=3)
            if frame is not None:
                filename = f"best3_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] Saved: {filename}")
                cv2.imshow("Best Frame (of 3)", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif choice == "4":
            frame = get_best_frame(n=5)
            if frame is not None:
                filename = f"best5_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] Saved: {filename}")
                cv2.imshow("Best Frame (of 5)", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif choice == "0":
            print("[INFO] Exiting")
            break

        else:
            print("[INFO] Invalid choice")