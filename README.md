---Factory Vision---
AI-powered smart factory monitoring and analytics system
Factory Vision is an intelligent automation system designed to monitor factory operations using Computer Vision, IoT sensors, and AI-driven analytics. It helps industries track production, detect defects in real time, ensure worker safety, and reduce energy waste using deep learning and data insights.

--Key Features
1. Real-time Object and Defect Detection

Uses YOLOv8 for detecting products, defects, and anomalies

Supports live camera feed or uploaded videos/images

Customizable for different factory environments

2. Production Line Monitoring

Tracks item count, throughput, bottlenecks, and delays

Automatically generates logs and inference results

3. IoT and Sensor Integration

Supports temperature, vibration, humidity, and other sensors

Alerts for abnormal readings or safety risks

4. Worker Safety Monitoring

Detects PPE compliance and unsafe behaviors

Identifies intrusion into restricted zones

5. Analytics and Reporting

Provides real-time metrics for production, defects, and energy usage

Supports exporting reports for factory management

--Tech Stack
-Computer Vision & AI

YOLOv8

PyTorch

OpenCV

-Backend & Automation

Python

Flask or FastAPI

Docker support

-IoT Integration

MQTT or HTTP sensors

Edge devices for on-site inference

-Data Handling

NumPy, Pandas

CSV, JSON log storage

--Project Structure
Factory-Vision/
│
├── data/                 # Training data
├── dataset/              # Dataset YAML & preprocessing scripts
├── inference/            # Output of inference runs
├── inference_results/    # Logs and result images/videos
├── logs/                 # System logs
├── runs/                 # YOLOv8 training and inference runs
├── scripts/              # Automation scripts
├── simline/              # Simulation or sample line data
│
├── app.py                # Main application entry point
├── train_model.py        # Training pipeline
├── run_infer.py          # Inference script
├── requirements.txt      # Dependencies
├── Dockerfile.txt        # Optional Docker support
└── yolov8n.pt            # Model weights

--How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run inference
python run_infer.py

3. Start the application
python app.py

--Future Improvements

Web analytics dashboard

Automated reporting

Multi-camera integration

Cloud deployment (AWS, Azure, GCP)

Edge-optimized lightweight model

--License

This project is for educational and research purposes.
