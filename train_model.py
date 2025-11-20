from ultralytics import YOLO
import time
from datetime import datetime
import csv

start_time = time.time()
start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

model = YOLO("yolov8n.pt")  # You can upgrade to yolov8s.pt or yolov8m.pt

model.train(
    data="dataset.yaml",
    epochs=30,         # You can increase to 50+ now
    imgsz=640,         # Or try 416 for speed, 768 for accuracy
    batch=16,          # Your 8GB VRAM can handle this easily
    device=0,          # 👈 Enables GPU training
    workers=2,         # Avoid CPU bottlenecks
    cache=True,        # Speeds up data loading
    val=True           # Keep validation for performance tracking
)

end_time = time.time()
end_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
duration = round(end_time - start_time, 2)

# Log training metrics
log_path = "training_log.csv"
log_entry = [start_stamp, end_stamp, duration, model.metrics.box.map50, model.metrics.box.map]
header = ["Start Time", "End Time", "Duration (s)", "mAP@0.5", "mAP@0.5:0.95"]

try:
    with open(log_path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(log_entry)
except FileExistsError:
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_entry)

print(f"Training completed in {duration} seconds. Log saved to {log_path}.")
