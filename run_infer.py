from ultralytics import YOLO
import os

# Load your trained model
model = YOLO("runs/detect/train3/weights/best.pt")  # Update path if needed

# Folder of images to run inference on
image_folder = "dataset/images/val"

# Output folder
output_folder = "inference_results"
os.makedirs(output_folder, exist_ok=True)

# Run inference on each image
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        results = model.predict(image_path, save=True, save_txt=True, project=output_folder, name="preds", exist_ok=True)
        print(f"Inferred: {filename}")
