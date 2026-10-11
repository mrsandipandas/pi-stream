import os
import socket
import struct
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO

app = Flask(__name__)

PI_HOST = os.environ.get("PI_HOST", "sandipan-pi.local")
PI_PORT = int(os.environ.get("PI_PORT", "9000"))
ROTATE_180 = os.environ.get("ROTATE_180", "1") == "1"
COUNT_AXIS = os.environ.get("COUNT_AXIS", "x")  # "x" = left/right, "y" = top/bottom
LINE_POS = float(os.environ.get("LINE_POS", "0.5"))  # 0-1, fraction of frame
DETECT_EVERY = int(os.environ.get("DETECT_EVERY", "1"))  # run YOLO every N frames

latest_frame = None
frame_lock = threading.Lock()
processed_frame = None
processed_lock = threading.Lock()

model = YOLO("yolov8n.pt")
PERSON_CLASS = 0

# --- Occupancy state ---
count_in = 0
count_out = 0
count_lock = threading.Lock()


MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.45"))
MAX_DIST = int(os.environ.get("MAX_DIST", "120"))


class CentroidTracker:
    def __init__(self, max_disappeared=10, confirm_frames=2):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.seen_count = OrderedDict()
        self.max_disappeared = max_disappeared
        self.confirm_frames = confirm_frames

    def register(self, centroid):
        oid = self.next_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.seen_count[oid] = 1
        self.next_id += 1
        return oid

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.seen_count[object_id]

    def confirmed_objects(self):
        return {oid: c for oid, c in self.objects.items()
                if self.seen_count[oid] >= self.confirm_frames}

    def update(self, centroids):
        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.confirmed_objects()

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
            return self.confirmed_objects()

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        obj_arr = np.array(object_centroids)
        new_arr = np.array(centroids)
        dists = np.linalg.norm(obj_arr[:, None] - new_arr[None, :], axis=2)

        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > MAX_DIST:
                continue
            oid = object_ids[row]
            self.objects[oid] = centroids[col]
            self.disappeared[oid] = 0
            self.seen_count[oid] += 1
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        unused_cols = set(range(len(centroids))) - used_cols

        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in unused_cols:
            self.register(centroids[col])

        return self.confirmed_objects()


tracker = CentroidTracker()
prev_positions = {}


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Lost connection")
        buf += chunk
    return buf


def receive_from_pi():
    global latest_frame
    while True:
        print(f"Connecting to Pi at {PI_HOST}:{PI_PORT}...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((PI_HOST, PI_PORT))
            print("Connected to Pi!")
            while True:
                header = recv_exact(sock, 4)
                size = struct.unpack(">I", header)[0]
                data = recv_exact(sock, size)
                with frame_lock:
                    latest_frame = data
        except (ConnectionRefusedError, ConnectionResetError, ConnectionError, OSError) as e:
            print(f"Connection failed: {e}, retrying in 3s...")
            time.sleep(3)


def detection_loop():
    global count_in, count_out, processed_frame
    frame_count = 0
    last_boxes = []

    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.02)
            continue

        arr = np.frombuffer(frame, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            time.sleep(0.02)
            continue

        if ROTATE_180:
            img = cv2.rotate(img, cv2.ROTATE_180)

        small_h, small_w = img.shape[:2]
        use_x = COUNT_AXIS == "x"

        # run YOLO on the small image
        if frame_count % DETECT_EVERY == 0:
            results = model(img, verbose=False, classes=[PERSON_CLASS], imgsz=320)[0]
            last_boxes = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONFIDENCE:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                last_boxes.append((x1, y1, x2, y2, conf))
        frame_count += 1

        # upscale for display
        scale_x = 640 / small_w
        scale_y = 480 / small_h
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        h, w = 480, 640
        line_px = int(LINE_POS * (w if use_x else h))

        # draw line
        if use_x:
            cv2.line(img, (line_px, 0), (line_px, h), (0, 255, 255), 2)
        else:
            cv2.line(img, (0, line_px), (w, line_px), (0, 255, 255), 2)

        # scale boxes to display size and collect centroids
        centroids = []
        scaled_boxes = []
        for x1, y1, x2, y2, conf in last_boxes:
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
            cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
            centroids.append((cx, cy))
            scaled_boxes.append((sx1, sy1, sx2, sy2, conf))

        for sx1, sy1, sx2, sy2, conf in scaled_boxes:
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.0%}", (sx1, sy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        objects = tracker.update(centroids)

        with count_lock:
            for oid, (cx, cy) in objects.items():
                pos = cx if use_x else cy
                if oid in prev_positions:
                    prev = prev_positions[oid]
                    if prev < line_px and pos >= line_px:
                        count_in += 1
                    elif prev > line_px and pos <= line_px:
                        count_out += 1
                prev_positions[oid] = pos

            for oid in list(prev_positions.keys()):
                if oid not in objects:
                    del prev_positions[oid]

            occupancy = max(0, count_in - count_out)

        for oid, (cx, cy) in objects.items():
            cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(img, f"ID {oid}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(img, f"IN: {count_in}  OUT: {count_out}  OCCUPANCY: {occupancy}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        _, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with processed_lock:
            processed_frame = out.tobytes()


def generate_mjpeg():
    while True:
        with processed_lock:
            frame = processed_frame
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Room Occupancy</title>
    <style>
        body { background: #111; color: #fff; font-family: sans-serif;
               display: flex; flex-direction: column; align-items: center;
               margin: 0; padding: 20px; }
        h1 { margin-bottom: 8px; }
        .stats { display: flex; gap: 24px; margin: 16px 0; font-size: 20px; }
        .stat { padding: 12px 24px; border-radius: 8px; text-align: center; }
        .stat-in { background: #1a4d1a; }
        .stat-out { background: #4d1a1a; }
        .stat-occ { background: #1a1a4d; }
        .stat .num { font-size: 36px; font-weight: bold; }
        img { max-width: 90vw; border: 2px solid #333; border-radius: 8px; }
        .info { color: #888; margin-top: 12px; font-size: 14px; }
    </style>
    <script>
        function updateStats() {
            fetch('/stats').then(r => r.json()).then(d => {
                document.getElementById('in').textContent = d.in;
                document.getElementById('out').textContent = d.out;
                document.getElementById('occ').textContent = d.occupancy;
            });
            setTimeout(updateStats, 1000);
        }
        window.onload = updateStats;
    </script>
</head>
<body>
    <h1>Room Occupancy Monitor</h1>
    <div class="stats">
        <div class="stat stat-in"><div>IN</div><div class="num" id="in">0</div></div>
        <div class="stat stat-out"><div>OUT</div><div class="num" id="out">0</div></div>
        <div class="stat stat-occ"><div>OCCUPANCY</div><div class="num" id="occ">0</div></div>
    </div>
    <img src="/stream" alt="Camera">
    <p class="info">Yellow line = counting boundary | Left→Right = IN | Right→Left = OUT</p>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/stream")
def stream():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with count_lock:
        return jsonify({"in": count_in, "out": count_out,
                        "occupancy": max(0, count_in - count_out)})


@app.route("/reset", methods=["POST"])
def reset():
    global count_in, count_out
    with count_lock:
        count_in = 0
        count_out = 0
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    threading.Thread(target=receive_from_pi, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)
