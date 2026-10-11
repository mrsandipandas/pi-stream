# Pi Camera — Room Occupancy Monitor

Streams video from a Raspberry Pi camera, detects people with YOLOv8, and tracks room occupancy by counting crossings over a configurable line.

## Architecture

- **Raspberry Pi** runs `pi_camera.py` — captures frames and serves them on port 9000
- **Docker container** runs `app.py` — connects to the Pi, runs YOLO detection, tracks people, serves an MJPEG stream with a dashboard at `http://localhost:8080`

## Setup

### Pi side

1. Copy the script and install dependencies:

```bash
scp pi_camera.py sandipan-pi@sandipan-pi.local:~/
ssh sandipan-pi@sandipan-pi.local
sudo apt update && sudo apt install -y python3-picamera2 python3-opencv
```

2. Set up auto-start:

```bash
sudo tee /etc/systemd/system/pi-camera.service << 'EOF'
[Unit]
Description=Pi Camera Stream
After=network-online.target
Wants=network-online.target

[Service]
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/python3 /home/sandipan-pi/pi_camera.py
Restart=always
RestartSec=3
User=sandipan-pi

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable pi-camera
sudo systemctl start pi-camera
```

### Docker side

```bash
docker build -t pi-stream .
docker run --rm -p 8080:8080 -e PI_HOST=sandipan-pi.local pi-stream
```

Open http://localhost:8080.

## Configuration

All flags are environment variables passed to `docker run` with `-e`:

| Flag | Default | Description |
|------|---------|-------------|
### Connection

| Flag | Default | Description |
|------|---------|-------------|
| `PI_HOST` | `sandipan-pi.local` | Hostname or IP of the Raspberry Pi |
| `PI_PORT` | `9000` | Port the Pi camera serves on |

### Image

| Flag | Default | Description |
|------|---------|-------------|
| `ROTATE_180` | `1` | Rotate image 180 degrees (`1` = on, `0` = off). Use when camera is mounted upside down |

### Counting line

A yellow line divides the frame. When a tracked person's centroid crosses the line, the system counts it as IN or OUT depending on direction.

| Flag | Default | Description |
|------|---------|-------------|
| `COUNT_AXIS` | `x` | Which axis the line crosses: `x` = vertical line (left/right movement), `y` = horizontal line (top/bottom movement) |
| `LINE_POS` | `0.5` | Position of counting line as fraction of frame (0.0 to 1.0). `0.5` = center, `0.3` = 30% from left/top |

- **Left → Right** (or Top → Bottom) = **IN**
- **Right → Left** (or Bottom → Top) = **OUT**
- **Occupancy** = IN − OUT (floored at 0)

### Detection and tracking

YOLOv8n runs person detection on each frame. A centroid tracker assigns stable IDs to detected people across frames to track their movement.

| Flag | Default | Description |
|------|---------|-------------|
| `MIN_CONFIDENCE` | `0.45` | Minimum YOLO confidence to count a detection as a person. Lower catches more (children, partial views) but risks false positives from furniture or shadows. Higher is stricter but may miss small or partially visible people |
| `MAX_DIST` | `80` | Maximum pixel distance to match a detection to an existing tracked person across frames. If a detection is further than this from any known person, it becomes a new ID. Keep proportional to frame size (80 is good for 320×240) |
| `DETECT_EVERY` | `1` | Run YOLO every N frames. At 2-5 fps from the Pi, keep at `1`. On faster cameras, `2` or `3` halves/thirds the CPU load by reusing bounding boxes between detections |

**Tuning guide:**

- Too many ghost IDs appearing → raise `MIN_CONFIDENCE` to `0.55` or `0.6`
- Missing children or crouching people → lower `MIN_CONFIDENCE` to `0.3`
- IDs swapping between nearby people → lower `MAX_DIST` to `50`
- Person loses their ID when briefly occluded → raise `MAX_DIST` to `100`
- A new person must be detected in **2 consecutive frames** before they get a confirmed ID (not configurable, hardcoded to reduce flicker)

Example with custom settings:

```bash
docker run --rm -p 8080:8080 \
  -e PI_HOST=192.168.1.50 \
  -e ROTATE_180=0 \
  -e COUNT_AXIS=y \
  -e LINE_POS=0.4 \
  -e MIN_CONFIDENCE=0.35 \
  -e MAX_DIST=60 \
  pi-stream
```

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard with live video and counters |
| `/stream` | GET | Raw MJPEG stream |
| `/stats` | GET | JSON: `{"in": N, "out": N, "occupancy": N}` |
| `/reset` | POST | Reset all counters to zero |

Reset counters: `curl -X POST http://localhost:8080/reset`

## Updating code

After editing `pi_camera.py`:

```bash
scp pi_camera.py sandipan-pi@sandipan-pi.local:~/
ssh sandipan-pi@sandipan-pi.local "sudo systemctl restart pi-camera"
```

After editing `app.py` (fast — no rebuild):

```bash
docker run --rm -p 8080:8080 -e PI_HOST=sandipan-pi.local \
  -v ~/Desktop/Sandipan/pi-stream/app.py:/app/app.py pi-stream
```

This mounts your local `app.py` into the container. Edit the file, restart the container, changes apply instantly. Only rebuild (`docker build -t pi-stream .`) when you change `requirements.txt` or `Dockerfile`.

## Debugging

### Pi camera not streaming

```bash
ssh sandipan-pi@sandipan-pi.local
sudo systemctl status pi-camera
sudo journalctl -u pi-camera -n 30
```

Common issues:
- `ModuleNotFoundError: picamera2` — run `sudo apt install -y python3-picamera2`
- Camera not detected — check the ribbon cable is seated properly, run `vcgencmd get_camera`

### Docker can't connect to Pi

- Check Pi is reachable: `ping sandipan-pi.local`
- Check Pi is serving: `nc -zv sandipan-pi.local 9000`
- Both devices must be on the same network

### Stream loads but no image

- Docker logs show "Connecting to Pi" but not "Connected" — Pi camera service isn't running
- Docker logs show "Connected" — frames are arriving but may be corrupt; check Pi camera logs

### Wi-Fi issues

The Pi B1+ has no built-in Wi-Fi — it uses a USB Wi-Fi dongle (2.4 GHz only). Config file:

```bash
cat /etc/wpa_supplicant/wpa_supplicant.conf
```

To change network:

```bash
sudo tee /etc/wpa_supplicant/wpa_supplicant.conf << 'EOF'
country=SE
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YOUR_SSID"
    psk="YOUR_PASSWORD"
}
EOF
sudo reboot
```
