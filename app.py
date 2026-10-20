# app.py
import os
import time
import uuid
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
from PIL import Image

# ---------------------
# Page + sidebar config
# ---------------------
st.set_page_config(page_title="Factory Vision — Real-time Quality Inspection", layout="wide", page_icon="🧪")
st.title("Factory Vision — Real-time Quality Inspection")
st.caption("Upload • Webcam • Simulated line • Analytics • CSV logs")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", "runs/detect/train3/weights/best.pt")
    conf_th = st.slider("Confidence threshold", 0.05, 0.9, 0.35, 0.01)
    iou_th = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.5, 0.05)
    imgsz = st.select_slider("Image size", options=[320, 416, 512, 640], value=640)
    sim_dir = st.text_input("Simulated line folder", "dataset/images/val")
    sim_delay = st.slider("Sim delay (s)", 0.1, 2.0, 0.6, 0.1)
    st.caption("Tip: point to your trained weights. Use 640 for best accuracy; 320 for speed.")

# ---------------------
# Caching & utilities
# ---------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return YOLO(path)

def decode_image_from_upload(uploaded) -> np.ndarray:
    if hasattr(uploaded, "getvalue"):
        data = uploaded.getvalue()
    else:
        data = uploaded.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def predict(model, img_bgr, conf=0.35, iou=0.5, size=640):
    res = model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=size, verbose=False)
    return res[0]

def draw_detections(img_bgr, result, names, color=(0, 0, 255)):
    out = img_bgr.copy()
    if result.boxes is None:
        return out, []
    dets = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        dets.append({"cls": names[cls], "conf": conf, "xyxy": (x1, y1, x2, y2)})
    return out, dets

def severity_from_dets(dets, img_shape):
    if not dets:
        return "none"
    H, W = img_shape[:2]
    scores = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        area = ((x2 - x1) * (y2 - y1)) / max(1, W * H)
        scores.append(area * d["conf"])
    s = sum(scores)
    if s < 0.002:
        return "low"
    elif s < 0.01:
        return "medium"
    else:
        return "high"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(LOG_DIR, "events.csv")

def append_log(image_source: str, image_path: str, dets, severity: str):
    ts = datetime.utcnow().isoformat()
    rows = []
    if dets:
        for d in dets:
            rows.append({
                "timestamp": ts,
                "source": image_source,
                "image_path": image_path,
                "cls": d["cls"],
                "conf": d["conf"],
                "x1": d["xyxy"][0], "y1": d["xyxy"][1], "x2": d["xyxy"][2], "y2": d["xyxy"][3],
                "severity": severity,
                "is_defect": 1
            })
    else:
        rows.append({
            "timestamp": ts, "source": image_source, "image_path": image_path,
            "cls": "OK", "conf": 1.0, "x1": 0, "y1": 0, "x2": 0, "y2": 0,
            "severity": "none", "is_defect": 0
        })
    df = pd.DataFrame(rows)
    if not os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, index=False)
    else:
        df.to_csv(LOG_CSV, mode="a", header=False, index=False)

def save_frame(img_bgr, subdir="data"):
    os.makedirs(subdir, exist_ok=True)
    fn = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(subdir, fn)
    cv2.imwrite(path, img_bgr)
    return path

try:
    model = load_model(model_path)
    model_names = model.names
except Exception as e:
    st.error(f"Failed to load model at '{model_path}': {e}")
    st.stop()

tab_live, tab_sim, tab_analytics = st.tabs(["Live inspection", "Simulated line", "Analytics"])

with tab_live:
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.radio("Input source", ["Upload image", "Webcam (snapshot)"], horizontal=True)
        if mode == "Upload image":
            uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
            if uploaded:
                img = decode_image_from_upload(uploaded)
                result = predict(model, img, conf_th, iou_th, imgsz)
                vis, dets = draw_detections(img, result, model_names)
                sev = severity_from_dets(dets, img.shape)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Severity: {sev}")
                img_path = save_frame(img, subdir="data/uploads")
                append_log("upload", img_path, dets, sev)
        else:
            cam = st.camera_input("Capture a snapshot")
            if cam:
                img = decode_image_from_upload(cam)
                result = predict(model, img, conf_th, iou_th, imgsz)
                vis, dets = draw_detections(img, result, model_names, color=(0, 255, 0))
                sev = severity_from_dets(dets, img.shape)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Severity: {sev}")
                img_path = save_frame(img, subdir="data/webcam")
                append_log("webcam", img_path, dets, sev)

    with col2:
        st.subheader("Quick stats (labels-based demo)")
        def read_val_label_counts():
            import glob
            cls_counts = []
            for p in glob.glob("dataset/labels/val/*.txt"):
                with open(p, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        cls_id = int(line.split()[0])
                        cls_counts.append(cls_id)
            if not cls_counts:
                return pd.DataFrame({"class": [], "counts": [], "class_name": []})
            s = pd.Series(cls_counts).value_counts().rename_axis("class").reset_index(name="counts")
            s["class_name"] = s["class"].map({0: "scratch", 1: "dent"})
            return s
        st.dataframe(read_val_label_counts(), use_container_width=True)

        st.subheader("Export last analytics run")
        if os.path.exists(LOG_CSV):
            df_dl = pd.read_csv(LOG_CSV)
            st.download_button("Download event logs CSV", data=df_dl.to_csv(index=False), file_name="events.csv", mime="text/csv")
        else:
            st.caption("No logs yet — run a prediction to create logs.")

with tab_sim:
    st.write("Place images in the folder set in the sidebar (default: dataset/images/val).")
    start = st.button("Start simulation")
    placeholder = st.empty()
    if start:
        if not os.path.isdir(sim_dir):
            st.error(f"Folder not found: {sim_dir}")
        else:
            files = [os.path.join(sim_dir, f) for f in sorted(os.listdir(sim_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for path in files:
                img = cv2.imread(path)
                result = predict(model, img, conf_th, iou_th, imgsz)
                vis, dets = draw_detections(img, result, model_names)
                sev = severity_from_dets(dets, img.shape)
                placeholder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True,
                                  caption=f"{os.path.basename(path)} | Severity: {sev}")
                append_log("simulated", path, dets, sev)
                time.sleep(sim_delay)

with tab_analytics:
    if os.path.exists(LOG_CSV):
        df_log = pd.read_csv(LOG_CSV)
        st.metric("Images inspected", df_log["image_path"].nunique())
        st.bar_chart(df_log["cls"].value_counts())
        st.dataframe(df_log.tail(50), use_container_width=True)
    else:
        st.caption("No logs yet — run a prediction to create logs.")
