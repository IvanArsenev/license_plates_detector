# 🚗 License Plate Detection and Training using YOLO

This project provides a robust solution for license plate detection and model training using YOLO (You Only Look Once) object detection framework. It supports both real-time detection on video streams and custom model training for license plate recognition.

## 🌟 Features
- **Real-time License Plate Detection**: Track and count unique license plates in video streams
- **Persistent Object Tracking**: Maintains plate IDs across frames with configurable memory
- **Training Pipeline**: Customizable YOLO model training with configurable parameters
- **Visualization**: Annotated video output with bounding boxes and plate IDs
- **Performance Metrics**: Logs frame-by-frame statistics and total unique plate count

## 🛠️ Installation
0. Install CUDA to run train in GPU
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/***
```
*** - version of CUDA (12.8 = 128)

1. Clone the repository:
```
git clone https://github.com/IvanArsenev/license_plates_detector
cd license_plates_detector
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## 📦 Project Structure
- `yolo_model.py`: Core implementation with YoloLicensePlateDetector class
- `README.md`: Project documentation
- `requirements.txt`: Python dependencies
- `data.yaml`: Config for Yolo train

## 🚀 Usage

### Demo Mode (Video Processing)
```
python yolo_model.py --mode demo --model_path ./results/yolo_training/weights/best.pt --license_plate_class 0 --video_path ./videos.mp4
```

### Training Mode
```
python yolo_model.py --mode train --model_path yolov8n.pt --config_path data.yaml --results_path results --epochs 150 --batch_size 16 --img_size 720
```

## ⚙️ Parameters

### Demo Mode
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Path to YOLO model file | Required |
| `--license_plate_class` | Class ID for license plates | 0 |
| `--video_path` | Path to input video file | Required |

### Training Mode
| Parameter | Description | Default   |
|-----------|-------------|-----------|
| `--model_path` | Path to base YOLO model | Required  |
| `--config_path` | Path to data.yaml config | Required  |
| `--results_path` | Output directory for results | "results" |
| `--epochs` | Number of training epochs | 150       |
| `--batch_size` | Training batch size | 16        |
| `--img_size` | Input image size | 720       |

## 📊 Output
- **Demo Mode**: Real-time annotated video with:
  - Bounding boxes around detected plates
  - Unique plate IDs
  - Console logging of detection statistics

- **Training Mode**: Saved model weights and training artifacts in specified results directory

## 📝 Example Output
```
2023-11-15 14:30:45,123 - INFO - New license plate detected! Total count: 5
2023-11-15 14:30:45,456 - INFO - Current frame: 90, Unique plates detected: 5
2023-11-15 14:31:22,789 - INFO - Final count of unique license plates: 8
```

## 🖼️ Dataset
Open dataset to download: `https://app.roboflow.com/etbx/license_plates-fepuu/1`

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📜 License
[MIT License](LICENSE)