import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from flask import Flask, Response, render_template_string, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)
app.json.sort_keys = False

# "pi"    = pull frames from the Raspberry Pi over TCP (original behaviour)
# "phone" = receive frames pushed from a phone browser at /phone (served over HTTPS)
SOURCE = os.environ.get("SOURCE", "pi").lower()
PI_HOST = os.environ.get("PI_HOST", "sandipan-pi.local")
PI_PORT = int(os.environ.get("PI_PORT", "9000"))
# Phone is usually held upright, so default to no rotation when it's the source.
ROTATE_180 = os.environ.get("ROTATE_180", "0" if SOURCE == "phone" else "1") == "1"
COUNT_AXIS = os.environ.get("COUNT_AXIS", "x")  # "x" = left/right, "y" = top/bottom
LINE_POS = float(os.environ.get("LINE_POS", "0.5"))  # 0-1, fraction of frame
REVERSE_DIRECTION = os.environ.get("REVERSE_DIRECTION", "0") == "1"  # 1 = right→left / bottom→top is IN
DETECT_EVERY = int(os.environ.get("DETECT_EVERY", "1"))  # run YOLO every N frames

latest_frame = None
last_frame_time = 0.0  # epoch seconds of the most recent ingested/received frame
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
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.2"))
MAX_AGE = int(os.environ.get("MAX_AGE", "15"))


def iou_batch(bb_test, bb_gt):
    x1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0:1].T)
    y1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1:2].T)
    x2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2:3].T)
    y2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_b = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # state: [cx, cy, area, aspect, vx, vy, va]
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1
        self.kf.transitionMatrix[1, 5] = 1
        self.kf.transitionMatrix[2, 6] = 1
        self.kf.measurementMatrix = np.zeros((4, 7), dtype=np.float32)
        self.kf.measurementMatrix[:4, :4] = np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32)
        self.kf.processNoiseCov[4:, 4:] *= 0.01
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.statePost = np.array(
            [[cx], [cy], [w * h], [w / (h + 1e-6)], [0], [0], [0]],
            dtype=np.float32)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.correct(np.array(
            [[cx], [cy], [w * h], [w / (h + 1e-6)]], dtype=np.float32))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        s = self.kf.statePost
        cx, cy, area, aspect = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        w = np.sqrt(max(area * aspect, 1))
        h = max(area / (w + 1e-6), 1)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def centroid(self):
        s = self.kf.statePost
        return int(s[0]), int(s[1])


class SORTTracker:
    def __init__(self, max_age=15, iou_threshold=0.2):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        # predict new locations
        for t in self.trackers:
            t.predict()

        if len(self.trackers) == 0:
            for d in detections:
                self.trackers.append(KalmanBoxTracker(d))
            return {t.id: t.centroid() for t in self.trackers}

        if len(detections) == 0:
            self.trackers = [t for t in self.trackers
                             if t.time_since_update <= self.max_age]
            return {t.id: t.centroid() for t in self.trackers
                    if t.time_since_update == 0}

        trk_boxes = np.array([t.get_state() for t in self.trackers])
        det_boxes = np.array(detections)
        iou_matrix = iou_batch(det_boxes, trk_boxes)

        matched_det = set()
        matched_trk = set()

        if iou_matrix.size > 0:
            row_idx, col_idx = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_idx, col_idx):
                if iou_matrix[r, c] >= self.iou_threshold:
                    self.trackers[c].update(detections[r])
                    matched_det.add(r)
                    matched_trk.add(c)

        for d in range(len(detections)):
            if d not in matched_det:
                self.trackers.append(KalmanBoxTracker(detections[d]))

        self.trackers = [t for i, t in enumerate(self.trackers)
                         if i in matched_trk or t.time_since_update <= self.max_age]

        return {t.id: t.centroid() for t in self.trackers}


tracker = SORTTracker(max_age=MAX_AGE, iou_threshold=IOU_THRESHOLD)
prev_positions = {}
first_positions = {}
counted_ids = set()


def get_lan_ip():
    """Best-effort local IP. Inside Docker (bridge network) this is the container
    IP, not the host LAN IP, so it's only a hint — use the IP the phone can reach."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def make_qr_svg(data, box=6):
    """Render a QR code for `data` as a self-contained SVG string (no deps at
    render time beyond `qrcode`). Returns "" if generation fails."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=2, box_size=1,
                           error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(data)
        qr.make(fit=True)
        matrix = qr.get_matrix()
    except Exception as e:
        print(f"QR generation failed: {e}")
        return ""
    n = len(matrix)
    size = n * box
    rects = "".join(
        f'<rect x="{x * box}" y="{y * box}" width="{box}" height="{box}"/>'
        for y, row in enumerate(matrix)
        for x, val in enumerate(row) if val
    )
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
            f'height="{size}" viewBox="0 0 {size} {size}" '
            f'shape-rendering="crispEdges"><rect width="{size}" height="{size}" '
            f'fill="#fff"/><g fill="#000">{rects}</g></svg>')


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
                    globals()["last_frame_time"] = time.time()
        except (ConnectionRefusedError, ConnectionResetError, ConnectionError, OSError) as e:
            print(f"Connection failed: {e}, retrying in 3s...")
            time.sleep(3)


def record_crossing(forward):
    # forward = left→right (x) or top→bottom (y); caller must hold count_lock
    global count_in, count_out
    if forward != REVERSE_DIRECTION:
        count_in += 1
    else:
        count_out += 1


def detection_loop():
    global processed_frame
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

        # scale boxes to display size
        scaled_boxes = []
        det_boxes = []
        for x1, y1, x2, y2, conf in last_boxes:
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
            scaled_boxes.append((sx1, sy1, sx2, sy2, conf))
            det_boxes.append([sx1, sy1, sx2, sy2])

        for sx1, sy1, sx2, sy2, conf in scaled_boxes:
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.0%}", (sx1, sy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        objects = tracker.update(det_boxes)

        with count_lock:
            for oid, (cx, cy) in objects.items():
                pos = cx if use_x else cy
                if oid not in first_positions:
                    first_positions[oid] = pos
                if oid in prev_positions:
                    prev = prev_positions[oid]
                    if prev < line_px and pos >= line_px:
                        record_crossing(forward=True)
                        counted_ids.add(oid)
                    elif prev > line_px and pos <= line_px:
                        record_crossing(forward=False)
                        counted_ids.add(oid)
                prev_positions[oid] = pos

            for oid in list(prev_positions.keys()):
                if oid not in objects:
                    if oid not in counted_ids and oid in first_positions:
                        first = first_positions[oid]
                        last = prev_positions[oid]
                        if first < line_px and last >= line_px:
                            record_crossing(forward=True)
                        elif first > line_px and last <= line_px:
                            record_crossing(forward=False)
                    del prev_positions[oid]
                    first_positions.pop(oid, None)
                    counted_ids.discard(oid)

            occupancy = max(0, count_in - count_out)

        for oid, (cx, cy) in objects.items():
            cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(img, f"ID {oid}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
        .join { background: #1b1b1b; border: 1px solid #333; border-radius: 10px;
                padding: 16px 20px; margin: 4px 0 8px; display: flex; gap: 20px;
                align-items: center; flex-wrap: wrap; justify-content: center; }
        .join .qr { background: #fff; padding: 8px; border-radius: 8px; line-height: 0; }
        .join .qr svg { width: 160px; height: 160px; display: block; }
        .join .txt { text-align: left; }
        .join .txt h2 { margin: 0 0 6px; font-size: 16px; }
        .join .txt a { color: #6cf; word-break: break-all; font-size: 15px; }
        .join .txt p { color: #999; font-size: 13px; margin: 6px 0 0; }
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
    <h1>Occupancy Monitor</h1>
    {% if show_join %}
    <div class="join">
        <div class="qr">{{ qr_svg | safe }}</div>
        <div class="txt">
            <h2>📱 Join from your phone</h2>
            <a href="{{ join_url }}">{{ join_url }}</a>
            <p>Scan the QR code, or type the link on your phone (same Wi-Fi). Accept the certificate warning, then tap Start.</p>
            {% if is_local %}<p style="color:#e88;">⚠ This page is open on <b>{{ host }}</b>. Reopen the dashboard at this machine's LAN IP so the QR points somewhere your phone can reach.</p>{% endif %}
        </div>
    </div>
    {% endif %}
    <div class="stats">
        <div class="stat stat-in"><div>IN</div><div class="num" id="in">0</div></div>
        <div class="stat stat-out"><div>OUT</div><div class="num" id="out">0</div></div>
        <div class="stat stat-occ"><div>OCCUPANCY</div><div class="num" id="occ">0</div></div>
    </div>
    <img src="/stream" alt="Camera">
    <p class="info">Yellow line = counting boundary | {{ in_dir }} = IN | {{ out_dir }} = OUT</p>
</body>
</html>
"""


@app.route("/")
def index():
    fwd, back = ("Left→Right", "Right→Left") if COUNT_AXIS == "x" else ("Top→Bottom", "Bottom→Top")
    if REVERSE_DIRECTION:
        fwd, back = back, fwd

    show_join = SOURCE == "phone"
    join_url = qr_svg = ""
    host = request.host
    is_local = host.split(":")[0] in ("localhost", "127.0.0.1")
    if show_join:
        join_url = f"https://{host}/phone"
        qr_svg = make_qr_svg(join_url)

    return render_template_string(
        HTML, in_dir=fwd, out_dir=back,
        show_join=show_join, join_url=join_url, qr_svg=qr_svg,
        host=host, is_local=is_local)


@app.route("/stream")
def stream():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with count_lock:
        return jsonify({"timestamp": time.time(),
                        "in": count_in, "out": count_out,
                        "occupancy": max(0, count_in - count_out)})


@app.route("/reset", methods=["POST"])
def reset():
    global count_in, count_out
    with count_lock:
        count_in = 0
        count_out = 0
    return jsonify({"status": "reset"})


# --- Phone-as-camera source ---------------------------------------------------

PHONE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Phone Camera</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #111; color: #fff; font-family: -apple-system, system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
  h1 { font-size: 18px; margin: 14px 0 4px; }
  video { width: 100%; max-width: 480px; background: #000; border-radius: 10px; margin-top: 8px; }
  .row { display: flex; gap: 10px; margin: 14px 0; flex-wrap: wrap; justify-content: center; }
  button { font-size: 16px; padding: 12px 20px; border: 0; border-radius: 8px; color: #fff; }
  #start { background: #1a7d1a; }
  #stop  { background: #7d1a1a; }
  #flip  { background: #333; }
  button:disabled { opacity: .4; }
  .stats { display: flex; gap: 20px; font-size: 14px; color: #bbb; margin-bottom: 8px; }
  .stats b { color: #fff; }
  #status { font-size: 14px; margin: 6px 16px; text-align: center; color: #9ad; min-height: 20px; }
</style>
</head>
<body>
  <h1>Phone Camera &rarr; Occupancy Monitor</h1>
  <div id="status">Tap Start and allow camera access.</div>
  <video id="v" autoplay playsinline muted></video>
  <div class="stats">
    <div>Sent: <b id="sent">0</b></div>
    <div>FPS: <b id="fps">0</b></div>
  </div>
  <div class="row">
    <button id="start">Start</button>
    <button id="stop" disabled>Stop</button>
    <button id="flip">Flip camera</button>
  </div>

<script>
const TARGET_W = 320, TARGET_H = 240, QUALITY = 0.6, MIN_INTERVAL_MS = 150;
const v = document.getElementById('v');
const statusEl = document.getElementById('status');
const sentEl = document.getElementById('sent');
const fpsEl = document.getElementById('fps');
const canvas = document.createElement('canvas');
canvas.width = TARGET_W; canvas.height = TARGET_H;
const ctx = canvas.getContext('2d');

let stream = null, running = false, facing = 'environment';
let sent = 0, lastTick = performance.now(), framesThisSec = 0;
let wakeLock = null;

function setStatus(msg, color) { statusEl.textContent = msg; statusEl.style.color = color || '#9ad'; }

async function startCamera() {
  try {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    v.srcObject = stream;
    await v.play();
    running = true;
    document.getElementById('start').disabled = true;
    document.getElementById('stop').disabled = false;
    setStatus('Streaming to monitor…', '#7d7');
    try { wakeLock = await navigator.wakeLock.request('screen'); } catch (e) {}
    pump();
  } catch (e) {
    setStatus('Camera error: ' + e.message + '. Make sure you opened this page over https and allowed the camera.', '#f88');
  }
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (wakeLock) { wakeLock.release(); wakeLock = null; }
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
  setStatus('Stopped.', '#9ad');
}

async function pump() {
  while (running) {
    const t0 = performance.now();
    try {
      ctx.drawImage(v, 0, 0, TARGET_W, TARGET_H);
      const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', QUALITY));
      if (blob) {
        const resp = await fetch('/ingest', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
        if (resp.ok) {
          sent++; framesThisSec++;
          sentEl.textContent = sent;
        } else {
          setStatus('Server responded ' + resp.status, '#f88');
        }
      }
    } catch (e) {
      setStatus('Send failed: ' + e.message, '#f88');
    }
    const now = performance.now();
    if (now - lastTick >= 1000) { fpsEl.textContent = framesThisSec; framesThisSec = 0; lastTick = now; }
    const elapsed = performance.now() - t0;
    if (elapsed < MIN_INTERVAL_MS) await new Promise(r => setTimeout(r, MIN_INTERVAL_MS - elapsed));
  }
}

document.getElementById('start').onclick = startCamera;
document.getElementById('stop').onclick = stopCamera;
document.getElementById('flip').onclick = () => { facing = (facing === 'environment') ? 'user' : 'environment'; if (running) startCamera(); };
// Re-acquire the wake lock if the tab is re-shown.
document.addEventListener('visibilitychange', async () => {
  if (running && document.visibilityState === 'visible' && !wakeLock) {
    try { wakeLock = await navigator.wakeLock.request('screen'); } catch (e) {}
  }
});
</script>
</body>
</html>
"""


@app.route("/phone")
def phone():
    return Response(PHONE_HTML, mimetype="text/html")


@app.route("/ingest", methods=["POST"])
def ingest():
    global latest_frame, last_frame_time
    data = request.get_data()
    if not data:
        return jsonify({"status": "empty"}), 400
    with frame_lock:
        latest_frame = data
        last_frame_time = time.time()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    if SOURCE == "pi":
        threading.Thread(target=receive_from_pi, daemon=True).start()
    else:
        ip = os.environ.get("HOST_IP") or get_lan_ip()
        print("=" * 64)
        print(f"  This machine IP : {ip}")
        print(f"  Dashboard + QR  : https://{ip}:8080/   (open this on your laptop)")
        print(f"  Phone camera    : https://{ip}:8080/phone")
        if not os.environ.get("HOST_IP"):
            print("  NOTE: in Docker this is the CONTAINER IP, which the phone cannot")
            print("        reach. Pass your host's LAN IP so this line is correct:")
            print("        docker run ... -e HOST_IP=$(ipconfig getifaddr en0) ...")
            print("        The on-screen QR is always correct as long as you open the")
            print("        dashboard at the host LAN IP (not localhost).")
        print("=" * 64)
    threading.Thread(target=detection_loop, daemon=True).start()

    # The phone camera (getUserMedia) only works in a secure context, so serve
    # over HTTPS with an on-the-fly self-signed cert when the phone is the source.
    ssl_context = "adhoc" if SOURCE == "phone" else None
    app.run(host="0.0.0.0", port=8080, threaded=True, ssl_context=ssl_context)
