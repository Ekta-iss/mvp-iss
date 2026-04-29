import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
ROOT = Path("../data")

FRAME_DIR = ROOT  / "raw" / "frames" / "video_01"

OUTPUT_DIR = ROOT / "annotations" / "yolo_labeled" / "video_01"

IMG_OUT = OUTPUT_DIR / "images"
LBL_OUT = OUTPUT_DIR / "labels"

# 🔥 CHANGE THIS TO YOUR TRAINED MODEL PATH
MODEL_PATH = Path("./runs/detect/runs/yolo_label_model/v1_cpu/weights/best.pt")

# Confidence threshold for auto-labeling
CONF_THRESH = 0.6

# =========================
# CREATE OUTPUT STRUCTURE
# =========================
def setup_dirs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    IMG_OUT.mkdir(parents=True, exist_ok=True)
    LBL_OUT.mkdir(parents=True, exist_ok=True)

# =========================
# AUTO-LABEL FUNCTION
# =========================
def auto_label():
    print("\n🚀 Loading model...\n")
    model = YOLO(str(MODEL_PATH))

    frames = list(FRAME_DIR.glob("*.jpg")) + list(FRAME_DIR.glob("*.png"))

    print(f"📊 Total frames found: {len(frames)}\n")

    for img_path in frames:
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            verbose=False
        )

        result = results[0]

        # Copy image to output folder
        shutil.copy(img_path, IMG_OUT / img_path.name)

        label_file = LBL_OUT / (img_path.stem + ".txt")

        with open(label_file, "w") as f:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                x_center, y_center, width, height = box.xywhn[0]

                # YOLO format: class x_center y_center width height
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

        print(f"✅ Labeled: {img_path.name}")

    print("\n🎉 Auto-labeling completed!")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("🚀 Starting auto-labeling pipeline...")

    setup_dirs()
    auto_label()

    print("\n✅ Done!")