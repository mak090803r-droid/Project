import cv2
import socket
import struct
import pickle

# 1. Setup Server Socket to receive frames
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999)) # Listen on all local channels
server_socket.listen(5)
print("📥 WINDOWS AI ENGINE ONLINE: Standing by for Linux Laptop video stream...")

client_socket, addr = server_socket.accept()
print(f"✔️ Connected to Linux Laptop Camera Node at: {addr}")

data = b""
payload_size = struct.calcsize("Q")

try:
    while True:
        # Retrieve packet size header
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet: break
            data += packet
            
        if not data: break
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        
        # Retrieve full image payload matrix
        while len(data) < msg_size:
            data += client_socket.recv(4096)
            
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        # Deserialize bytes back into an OpenCV image frame array
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        # =========================================================
        # 🔥 YOUR MASTER AI PIPELINE TRIPPERS RUN HERE LIVE
        # =========================================================
        # This frame variable is now a standard OpenCV image matrix.
        # pass it straight into: ocr_engine.ocr(frame)
        # =========================================================
        
        # Display the incoming live feed from your laptop
        cv2.imshow("Live Stream from Linux Laptop Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()