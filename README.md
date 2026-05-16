# 🚀 SmartVision Logistics MVP

## 📌 Overview

**SmartVision Logistics MVP** is an intelligent computer vision system for automated container and equipment detection, tracking, and logistics management in warehouse/port operations. The system leverages YOLOv11 for real-time object detection and multi-object tracking to monitor crane operations, spreader equipment, and container movements.

### 🎯 Key Features
- **Real-time Object Detection** - Detects 8 classes: cranes, spreaders, containers, hooks, AGVs, and lane markers
- **Multi-Object Tracking** - Tracks equipment and containers across video frames
- **Automated Labeling** - Auto-labels frames using trained YOLO models
- **Dataset Management** - Merges CVAT and YOLO annotations with watermark removal
- **Video Inference** - Visualizes detections with tracking IDs and logistics alerts
- **Performance Metrics** - Evaluates model accuracy with precision, recall, and mAP
- **Smart Logistics Logic** - Detects correct/incorrect container pickups and proximity warnings

---

## 📁 Project Structure

```
mvp-iss/
├── src/
│   ├── prepare_dataset_yolo.py              # Dataset preparation & splitting
│   ├── train_yolo_tolabel_and_evaluate.py   # Complete training pipeline
│   ├── auto_label_frames.py                 # Auto-label frames with trained model
│   ├── evaluate_yolo_model_metrics.py       # Model evaluation (metrics)
│   ├── evaluate_yolo_model_visual.py        # Model evaluation (visual)
│   ├── merge_cvat_yolo_dataset.py           # Merge CVAT & YOLO datasets
│   ├── train_yolo_model_alignment_local.py  # Local training variant 1
│   ├── train_yolo_model_alignment_local2.py # Local training variant 2
│   └── visualize_inference.py               # Real-time video inference with tracking
├── data/
│   ├── raw/                                 # Raw videos & frames
│   ├── processed/                           # Processed datasets
│   ├── annotations/                         # CVAT & YOLO annotations
│   └── debug_visualization/                 # Debug visualizations
├── runs/                                    # Training outputs (models, weights)
├── .gitignore                               # Git ignore rules
└── README.md                                # This file

```

---

## 📖 Module Descriptions

### **prepare_dataset_yolo.py**
Prepares and organizes the YOLO dataset for training.

**Features:**
- Loads image-label pairs from merged dataset
- Splits data into train/val/test (80/10/10)
- Removes watermarks and adjusts bounding box labels
- Creates debug visualizations
- Generates `data.yaml` configuration file

**Usage:**
```bash
python src/prepare_dataset_yolo.py
```

**Output:** `../data/processed/final_yolo_dataset1/`

---

### **train_yolo_tolabel_and_evaluate.py**
Complete end-to-end training pipeline.

**Steps:**
1. Creates YOLO directory structure
2. Splits dataset (train/val/test)
3. Copies files with proper labels
4. Creates `data.yaml`
5. Trains YOLOv11n (or loads existing model)
6. Evaluates on validation set

**Configuration:**
```python
SEED = 42
EPOCHS = 50
BATCH = 8
IMAGE_SIZE = 512
DEVICE = "cpu"
OPTIMIZER = "Adam"
```

**Usage:**
```bash
python src/train_yolo_tolabel_and_evaluate.py
```

**Output:** Model weights in `./runs/yolo_label_model/v1_cpu/weights/best.pt`

---

### **auto_label_frames.py**
Automatically labels video frames using a trained YOLO model.

**Features:**
- Loads trained YOLO model
- Processes all frames in a directory
- Generates labels with configurable confidence threshold
- Copies images and creates corresponding `.txt` label files

**Configuration:**
```python
MODEL_PATH = "./runs/detect/runs/yolo_label_model/v1_cpu/weights/best.pt"
CONF_THRESH = 0.6
```

**Usage:**
```bash
python src/auto_label_frames.py
```

**Output:**
- Images: `../data/annotations/yolo_labeled/video_01/images/`
- Labels: `../data/annotations/yolo_labeled/video_01/labels/`

---

### **visualize_inference.py**
Real-time video inference with object tracking and logistics visualization.

**Features:**
- Real-time YOLO detection with BoT-SORT tracking
- Tracks inventory across frames
- Highlights target container for pickup jobs
- Warns on incorrect container hovers
- Displays dashboard with job order and inventory count
- Color-coded boxes: blue (crane), yellow (spreader), green (container), orange (target), red (warning)

**Configuration:**
```python
TARGET_WORK_ORDER = 25  # Container ID to pick
MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"
VIDEO_PATH = "../data/raw/videos/video_01.mp4"
CONF_THRESHOLD = 0.25
```

**Usage:**
```bash
python src/visualize_inference.py
```

**Output:** Display window with annotated video (Press 'q' to quit)

---

### **evaluate_yolo_model_metrics.py**
Evaluates model performance with numerical metrics.

**Metrics:**
- Precision, Recall, F1-Score
- mAP (mean Average Precision)
- Per-class performance

**Usage:**
```bash
python src/evaluate_yolo_model_metrics.py
```

---

### **evaluate_yolo_model_visual.py**
Generates visual evaluation outputs.

**Outputs:**
- Prediction visualizations
- Confusion matrices
- Sample detections

**Usage:**
```bash
python src/evaluate_yolo_model_visual.py
```

---

### **merge_cvat_yolo_dataset.py**
Merges datasets from CVAT annotation tool with YOLO format.

**Features:**
- Converts CVAT XML annotations to YOLO format
- Consolidates multiple dataset sources
- Validates image-label pairs

**Usage:**
```bash
python src/merge_cvat_yolo_dataset.py
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU support)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Ekta-iss/mvp-iss.git
cd mvp-iss
git checkout smartVision1
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy pyyaml
```

### Optional: For GPU Support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Dataset Structure

Expected directory structure for input data:

```
data/
├── raw/
│   ├── frames/
│   │   └── video_01/
│   │       ├── frame_001.jpg
│   │       ├── frame_002.jpg
│   │       └── ...
│   └── videos/
│       └── video_01.mp4
├── annotations/
│   ├── cvat/
│   │   └── video_01/
│   │       ├── images/
│   │       └── labels/
│   └── yolo_labeled/
├── processed/
│   ├── cvat_yolo_merged_dataset/
│   │   ├── images/
│   │   └── labels/
│   └── final_yolo_dataset1/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
└── debug_visualization/
```

---

## 🏷️ Class Definitions

| ID | Class | Description |
|----|-------|-------------|
| 0 | **crane** | Overhead crane equipment |
| 1 | **spreader** | Container spreader bar |
| 2 | **spreader_corner** | Spreader corner/attachment point |
| 3 | **hook** | Lifting hook |
| 4 | **container** | Shipping container |
| 5 | **container_corner** | Container corner/attachment |
| 6 | **AGV** | Automated Guided Vehicle |
| 7 | **lane_marker** | Lane marking/guide |

---

## 🚀 Quick Start Workflow

### Step 1: Prepare Dataset
```bash
python src/prepare_dataset_yolo.py
```
This will:
- Validate image-label pairs
- Remove watermarks
- Split into train/val/test
- Create debug visualizations

### Step 2: Train Model
```bash
python src/train_yolo_tolabel_and_evaluate.py
```
This will:
- Create YOLOv11n model (or load existing)
- Train for 50 epochs
- Evaluate on validation set
- Save best weights

### Step 3: Auto-label New Frames (Optional)
```bash
python src/auto_label_frames.py
```
Auto-generates labels for unlabeled frames using the trained model.

### Step 4: Inference on Video
```bash
python src/visualize_inference.py
```
Watch real-time detection with tracking and logistics alerts!

---

## ⚙️ Configuration Guide

### Model Training Parameters (`train_yolo_tolabel_and_evaluate.py`)

```python
# Training
epochs=50                  # Number of training epochs
imgsz=512                 # Image size
batch=8                   # Batch size
patience=15               # Early stopping patience

# Augmentation
hsv_h=0.015
hsv_s=0.7
hsv_v=0.4
degrees=3.0
translate=0.05
scale=0.3
shear=1.0
mosaic=0.5

# Optimization
lr0=0.001                 # Learning rate
optimizer="Adam"

# Device
device="cpu"              # Change to 0 for GPU
workers=0
```

### Dataset Split (`prepare_dataset_yolo.py`)

```python
TRAIN_RATIO = 0.8        # 80% training
VAL_RATIO = 0.1          # 10% validation
TEST_RATIO = 0.1         # 10% testing
```

### Inference Configuration (`visualize_inference.py`)

```python
TARGET_WORK_ORDER = 25    # Target container ID
CONF_THRESHOLD = 0.25     # Detection confidence
LINE_THICKNESS = 3
FONT_SCALE = 0.6
```

---

## 🤖 Model Details

**Base Model:** YOLOv11 Nano (yolo11n.pt)

**Model Specs:**
- Parameters: ~2.6M
- Size: ~6.3 MB
- Speed: Real-time on CPU/GPU

**Training Environment:**
- Framework: Ultralytics YOLOv11
- Loss: Combination of box, class, and DFL loss
- Optimizer: Adam
- Device: CPU (configurable for GPU)

**Performance Expectations:**
- Inference Speed: ~20-50 ms per frame (CPU)
- Detection Speed: Real-time on GPU
- Model Accuracy: Improves with more labeled data

---

## 👁️ Visualization Features

### Dashboard (visualize_inference.py)
- **Top Left:** Current job order (Pick up Container #ID)
- **Top Right:** Inventory count (unique containers seen)
- **Bottom Center:** Active alerts (Target matched or wrong container)

### Color Coding
- 🔵 **Blue** - Crane
- 🟡 **Yellow** - Spreader
- 🟢 **Green** - Container
- 🟠 **Orange** - Target container (correct pickup)
- 🔴 **Red** - Warning (wrong container hovered)

### Tracking Logic
- Tracks unique IDs across frames using BoT-SORT
- Detects "hover" (spreader proximity < 85 pixels)
- Validates pickup against target work order
- Maintains inventory of seen containers

---

## 🔧 Troubleshooting

### 1. **"Module not found" errors**
```bash
# Ensure all dependencies are installed
pip install ultralytics opencv-python numpy pyyaml
```

### 2. **Model not found**
- Check `MODEL_PATH` in scripts
- Ensure training completed successfully
- Look in `./runs/yolo_label_model/v1_cpu/weights/best.pt`

### 3. **Video playback issues**
```bash
# Check video codec and format
pip install --upgrade opencv-python
```

### 4. **Out of memory during training**
```python
# Reduce batch size in train_yolo_tolabel_and_evaluate.py
batch=4  # Instead of 8
```

### 5. **Slow inference on CPU**
- Switch to GPU: Change `device="cpu"` to `device=0`
- Install CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### 6. **Missing dataset files**
- Verify data structure matches expected layout
- Check `.gitignore` (data/ files may not be in repo)
- Ensure label files match image names

---

## 💡 Use Cases

### Logistics Automation
Monitor container movements and automate work order validation in ports and warehouses.

### Quality Assurance
Track equipment usage and detect mishandling of containers during operations.

### Training & Compliance
Record and analyze crane operator behavior for safety training and compliance auditing.

### Inventory Management
Auto-generate inventory counts and track container locations throughout the facility.

### Process Optimization
Identify bottlenecks and optimize container movement workflows.

---

## 📝 License

This project is provided as-is for internal use.

---

## 👨‍💻 Author

**Ekta-iss**  
GitHub: [Ekta-iss](https://github.com/Ekta-iss)

---

## 📞 Support

For issues, questions, or contributions, please open a GitHub issue in the repository.

---

## 🎓 References

- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [BoT-SORT Tracking](https://github.com/NirAharon/BoT-SORT)
- [CVAT Annotation Tool](https://www.cvat.ai/)

---

**Last Updated:** May 16, 2026

