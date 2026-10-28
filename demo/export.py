"""Build the browser demo's files from the trained run.

    python demo/export.py       (from the repo root; needs ultralytics, onnx and onnxruntime)

demo/model.onnx     runs/detect/train3/weights/best.pt exported to ONNX, run in the page by ONNX Runtime Web
demo/model.json     class names and validation scores, read from runs/detect/train3/results.csv
demo/samples/       a few synthetic validation images (never the third-party photos)

The check at the end runs the ONNX file with the same pre- and post-processing as index.html and
compares the boxes with what the original weights predict through ultralytics.
"""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/detect/train3"
DEMO = ROOT / "demo"
SAMPLES = ["00001", "00002", "00012", "00015"]  # two dents, two scratches; 00000, 00011 and 00029 are third-party photos
SIZE, CONF, IOU = 640, 0.35, 0.5  # app.py's defaults


def export():
    model = YOLO(RUN / "weights/best.pt")
    onnx = Path(model.export(format="onnx", imgsz=SIZE, opset=17, simplify=True, dynamic=True))
    shutil.move(onnx, DEMO / "model.onnx")
    last = list(csv.DictReader(open(RUN / "results.csv", encoding="utf-8")))[-1]
    last = {k.strip(): v.strip() for k, v in last.items()}
    labels = [l.split()[0] for f in (ROOT / "dataset/labels/val").glob("*.txt") for l in f.read_text().splitlines() if l.strip()]
    meta = {"names": [model.names[i] for i in sorted(model.names)], "imgsz": SIZE, "conf": CONF, "iou": IOU,
            "epoch": int(float(last["epoch"])), "map50": round(float(last["metrics/mAP50(B)"]), 3),
            "map50_95": round(float(last["metrics/mAP50-95(B)"]), 3),
            "val_images": len(list((ROOT / "dataset/labels/val").glob("*.txt"))),
            "val_scratches": labels.count("0"), "val_dents": labels.count("1"),
            "samples": [f"samples/{s}.jpg" for s in SAMPLES]}
    (DEMO / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (DEMO / "samples").mkdir(exist_ok=True)
    for s in SAMPLES:
        shutil.copy(ROOT / f"dataset/images/val/{s}.jpg", DEMO / f"samples/{s}.jpg")
    return model


def letterbox(img):
    """Same as index.html and ultralytics: scale the long side to 640, pad on grey 114 up to the next
    multiple of 32 (a 640x480 image stays 640x480), RGB in 0..1, CHW."""
    w, h = img.size
    r = min(SIZE / w, SIZE / h)
    nw, nh = round(w * r), round(h * r)
    pw, ph = (SIZE - nw) % 32, (SIZE - nh) % 32
    px, py = round(pw / 2 - 0.1), round(ph / 2 - 0.1)
    canvas = Image.new("RGB", (nw + pw, nh + ph), (114, 114, 114))
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (px, py))
    return (np.asarray(canvas, np.float32) / 255).transpose(2, 0, 1)[None], r, px, py


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect(sess, img):
    """Same decoding as index.html: [1, 4 + classes, 8400] -> boxes in image pixels, per-class NMS."""
    x, r, px, py = letterbox(img)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    scores = out[4:]
    cls, conf = scores.argmax(0), scores.max(0)
    keep = []
    for i in np.argsort(-conf):
        if conf[i] < CONF:
            break
        cx, cy, w, h = out[:4, i]
        box = [min(max((v - p) / r, 0), lim) for v, p, lim in
               ((cx - w / 2, px, img.width), (cy - h / 2, py, img.height), (cx + w / 2, px, img.width), (cy + h / 2, py, img.height))]
        if all(k[0] != cls[i] or iou(k[2], box) <= IOU for k in keep):
            keep.append((int(cls[i]), float(conf[i]), box))
    return keep


def same(a, b, conf_tol, iou_min):
    a, b = sorted(a, key=lambda d: -d[1]), sorted(b, key=lambda d: -d[1])
    return len(a) == len(b) and all(c1 == c2 and abs(p1 - p2) < conf_tol and iou(b1, b2) > iou_min
                                    for (c1, p1, b1), (c2, p2, b2) in zip(a, b))


def check(model):
    """Our decoding vs ultralytics on the same ONNX file, and the ONNX file vs the original weights."""
    sess, onnx = ort.InferenceSession(str(DEMO / "model.onnx")), YOLO(DEMO / "model.onnx", task="detect")
    boxes = lambda r: [(int(c), float(p), b.tolist()) for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    n = 0
    for img in sorted((ROOT / "dataset/images/val").glob("*.jpg")):
        if img.stem in ("00000", "00011", "00029"):
            continue
        ours = detect(sess, Image.open(img))
        kw = dict(imgsz=SIZE, conf=CONF, iou=IOU, verbose=False)
        assert same(ours, boxes(onnx.predict(str(img), **kw)[0]), 0.005, 0.99), img.name
        assert same(ours, boxes(model.predict(str(img), **kw)[0]), 0.005, 0.99), img.name
        n += 1
    print(f"ok: the page's decoding matches ultralytics, and the ONNX file matches best.pt, on {n} synthetic validation images")


if __name__ == "__main__":
    check(export())
