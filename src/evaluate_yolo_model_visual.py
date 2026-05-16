from ultralytics import YOLO
import cv2
import os
import random
from glob import glob

# =========================
# CONFIG
# =========================
MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"

TEST_IMAGE_DIR = "../data/processed/final_yolo_dataset1/images/test"
OUTPUT_DIR = "./runs/eval_results_visual"

CONF_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)

# =========================
# CLASS COLORS
# =========================
CLASS_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (255, 255, 0),
    4: (0, 0, 255),
    5: (255, 0, 255),
    6: (128, 0, 255),
    7: (0, 165, 255),
}

# =========================
# LOAD TEST IMAGES
# =========================
image_paths = glob(os.path.join(TEST_IMAGE_DIR, "*.jpg"))
samples = random.sample(image_paths, min(10, len(image_paths)))

print(f"🔍 Evaluating on {len(samples)} test images")

# =========================
# DRAW FUNCTION
# =========================
def draw_boxes(img, result):
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls, (255, 255, 255))
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img

# =========================
# RUN INFERENCE
# =========================
for i, img_path in enumerate(samples):

    img = cv2.imread(img_path)

    results = model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    annotated = draw_boxes(img.copy(), results[0])

    save_path = os.path.join(OUTPUT_DIR, f"eval_{i}.jpg")
    cv2.imwrite(save_path, annotated)

    print(f"Saved: {save_path}")

print("\n✅ Evaluation complete. Results saved in:", OUTPUT_DIR)