from ultralytics import YOLO
import torch

# =========================
# CONFIG
# =========================

DATA_YAML = "../data/processed/final_yolo_dataset/data.yaml"

MODEL_WEIGHTS = "yolov8n.pt"   # ⚠️ use nano for CPU
EPOCHS = 40                   # keep lower for CPU
IMG_SIZE = 640
BATCH_SIZE = 4                # safer for CPU

PROJECT_DIR = "runs/detect"
EXPERIMENT_NAME = "alignment_v1"

# =========================
# TRAIN
# =========================

def train():
    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,              # ⚠️ important for Windows CPU
        optimizer="SGD",        # faster on CPU
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=10,
        cache=True,             # ⚠️ speeds up CPU training
        amp=False,              # no mixed precision on CPU
        verbose=True,

        # 🔥 IMPORTANT for your use-case
        close_mosaic=10,        # stabilize later training
        degrees=5,              # slight rotation
        scale=0.3,              # scale variation
        shear=0.0,
        perspective=0.0,

        # small object tuning (VERY IMPORTANT)
        box=7.5,                # increase box loss importance
        cls=0.5,
        dfl=1.5
    )

    return results

# =========================
# VALIDATION
# =========================

def validate():
    model = YOLO(f"{PROJECT_DIR}/{EXPERIMENT_NAME}/weights/best.pt")

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640
    )

    print("\n📊 Validation Metrics:")
    print(metrics)

# =========================

if __name__ == "__main__":
    train()
    validate()