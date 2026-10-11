"""Run this on the Raspberry Pi. It serves camera frames on port 9000.

Any client on the network can connect to receive the stream.

Install on Pi:
    sudo apt update && sudo apt install -y python3-picamera2 python3-opencv

Usage:
    python3 pi_camera.py
"""
import socket
import struct
import time

from picamera2 import Picamera2
import cv2

PORT = 9000

cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (320, 240), "format": "RGB888"}
))
cam.start()
time.sleep(1)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)
print(f"Camera streaming on port {PORT}, waiting for client...")

while True:
    conn, addr = server.accept()
    print(f"Client connected: {addr}")
    try:
        while True:
            frame = cam.capture_array()
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = jpeg.tobytes()
            conn.sendall(struct.pack(">I", len(data)) + data)
    except (BrokenPipeError, ConnectionResetError):
        print(f"Client disconnected: {addr}")
    finally:
        conn.close()
