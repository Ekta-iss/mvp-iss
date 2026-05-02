from ultralytics import YOLO
import torch
import os

# =========================
# CONFIG
# =========================
DATA_YAML = "../data/processed/final_yolo_dataset/data.yaml"
MODEL_WEIGHTS = "yolo11n.pt"   # pre-trained nano model
OUTPUT_DIR = "./runs/detect"

EPOCHS = 5
IMG_SIZE = 640
BATCH_SIZE = 4   # keep small for CPU

# =========================
# CHECK CPU
# =========================
device = "cpu"
print(f"🚀 Using device: {device}")

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(f"❌ Model weights not found: {MODEL_WEIGHTS}")

model = YOLO(MODEL_WEIGHTS)

# =========================
# TRAIN
# =========================
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=device,          # ✅ FORCE CPU
    project=OUTPUT_DIR,
    name="crane_alignment_model",
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",

    # 🔥 Let YOLO handle augmentation
    augment=True,

    # stability
    workers=2,
    verbose=True
)

# =========================
# SAVE FINAL MODEL
# =========================
best_model_path = os.path.join(
    OUTPUT_DIR,
    "crane_alignment_model",
    "weights",
    "best.pt"
)

print(f"\n✅ Training complete!")
print(f"📦 Best model saved at: {best_model_path}")

# Optional: load and save explicitly again
trained_model = YOLO(best_model_path)
trained_model.save("crane_alignment_2.pt")

print("💾 Final model exported as crane_alignment_2.pt")