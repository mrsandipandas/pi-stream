FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
COPY app.py .
EXPOSE 8080
CMD ["python", "app.py"]
