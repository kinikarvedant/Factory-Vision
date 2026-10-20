from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

# iou=0.5 matches app.py's NMS default and merges overlapping same-class boxes (Ultralytics default is 0.7).
# No exist_ok: save_txt appends to existing label files, so re-running into the same folder doubled every box.
# Each run now writes to a fresh inference_results/preds* folder. save_conf keeps scores for threshold tuning.
model.predict("dataset/images/val", iou=0.5, save=True, save_txt=True, save_conf=True,
              project="inference_results", name="preds")
