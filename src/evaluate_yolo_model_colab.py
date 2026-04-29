from ultralytics import YOLO
import os
import yaml

# =========================
# BASE PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "runs",
    "detect",
    "v1",
    "train-colab",
    "weights",
    "best.pt"
)

DATA_YAML_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_yolo_dataset",
    "data.yaml"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "src", "runs", "eval")

# =========================
# LOAD CLASS NAMES
# =========================
with open(DATA_YAML_PATH, "r") as f:
    data_config = yaml.safe_load(f)

CLASS_NAMES = data_config["names"]

# =========================
# VALIDATION CHECKS
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

if not os.path.exists(DATA_YAML_PATH):
    raise FileNotFoundError(f"❌ data.yaml not found: {DATA_YAML_PATH}")

print(f"✅ Model loaded from: {MODEL_PATH}")
print(f"✅ Dataset config: {DATA_YAML_PATH}")

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# FUNCTION: RUN EVALUATION
# =========================
def evaluate(split="val"):
    print(f"\n🚀 Running evaluation on: {split.upper()} set")

    metrics = model.val(
        data=DATA_YAML_PATH,
        split=split,          # 'val' or 'test'
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.5,
        save=True,
        save_json=True,
        project=OUTPUT_DIR,
        name=f"alignment_eval_{split}",
        verbose=True
    )

    print("\n📊 ===== OVERALL METRICS =====")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\n📦 ===== PER-CLASS AP50 =====")
    for i, ap in enumerate(metrics.box.ap50):
        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"{class_name:20s}: {ap:.4f}")

    return metrics


# =========================
# RUN BOTH VAL + TEST
# =========================
val_metrics = evaluate("val")
test_metrics = evaluate("test")

print("\n✅ Evaluation complete!")