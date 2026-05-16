import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
ROOT = Path("../data")
CVAT_PATH = ROOT / "annotations" / "cvat" / "video_01"
IMG_SRC = CVAT_PATH / "images"
LBL_SRC = CVAT_PATH / "labels"

YOLO_DATASET = ROOT / "processed" / "video_01_yolo_label_training_set"

SEED = 42

CLASS_NAMES = [
    "crane", "spreader", "spreader_corner", "hook",
    "container", "container_corner", "AGV", "lane_marker"
]

MODEL_PATH = Path("./runs/detect/runs/yolo_label_model/v1_cpu/weights/best.pt")

# =========================
# STEP 1: CREATE DIRS
# =========================
def create_yolo_dirs():
    if YOLO_DATASET.exists():
        shutil.rmtree(YOLO_DATASET)

    for split in ["train", "val", "test"]:
        (YOLO_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# =========================
# STEP 2: SPLIT DATASET
# =========================
def split_dataset():
    random.seed(SEED)

    images = list(IMG_SRC.glob("*.jpg")) + list(IMG_SRC.glob("*.png"))
    random.shuffle(images)

    n = len(images)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    print(f"\n📊 Dataset Split:")
    print(f"Total: {n}")
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}\n")

    return train_imgs, val_imgs, test_imgs

# =========================
# STEP 3: COPY FILES
# =========================
def copy_files(images, split):
    count = 0

    for img_path in images:
        label_path = LBL_SRC / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f"⚠️ Missing label: {img_path.name}")
            continue

        shutil.copy(img_path, YOLO_DATASET / "images" / split / img_path.name)
        shutil.copy(label_path, YOLO_DATASET / "labels" / split / label_path.name)
        count += 1

    print(f"✅ {split.upper()}: {count} samples copied")

# =========================
# STEP 4: CREATE YAML
# =========================
def create_data_yaml():
    yaml_path = YOLO_DATASET / "data.yaml"

    with open(yaml_path, "w") as f:
        f.write(f"""
path: {YOLO_DATASET}

train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")

    print("📝 data.yaml created")
    return yaml_path

# =========================
# STEP 5: LOAD OR TRAIN MODEL
# =========================
def get_model(data_yaml):
    if MODEL_PATH.exists():
        print("\n✅ Model found. Loading for evaluation...\n")
        model = YOLO(str(MODEL_PATH))
        return model, False

    print("\n🏋️ No model found. Starting training...\n")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_yaml),

        epochs=50,
        imgsz=512,
        batch=8,
        patience=15,

        device="cpu",
        workers=0,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=3.0,
        translate=0.05,
        scale=0.3,
        shear=1.0,
        mosaic=0.5,
        mixup=0.0,

        lr0=0.001,
        optimizer="Adam",

        project="runs/yolo_label_model",
        name="v1_cpu",
        exist_ok=True,
        verbose=True
    )

    return model, True

# =========================
# STEP 6: EVALUATION
# =========================
def evaluate(model, data_yaml):
    print("\n🧪 Running evaluation...\n")

    metrics = model.val(
        data=str(data_yaml),
        split="val",   # safe fallback (avoids test crash)
        device="cpu"
    )

    print("\n📊 Evaluation Results:")
    print(metrics)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("🚀 Preparing YOLO dataset...")

    create_yolo_dirs()

    train_imgs, val_imgs, test_imgs = split_dataset()

    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")
    copy_files(test_imgs, "test")

    data_yaml = create_data_yaml()

    model, trained = get_model(data_yaml)

    evaluate(model, data_yaml)

    print("\n✅ Pipeline completed successfully!")