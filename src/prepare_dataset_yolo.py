import os
import shutil
import random
from pathlib import Path
import cv2

# =========================
# CONFIG
# =========================
INPUT_IMAGE_DIR = "../data/processed/cvat_yolo_merged_dataset/images"
INPUT_LABEL_DIR = "../data/processed/cvat_yolo_merged_dataset/labels"

OUTPUT_DIR = "../data/processed/final_yolo_dataset1"
DEBUG_DIR = "../data/debug_visualization"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DEBUG = True
DEBUG_SAMPLES = 3

CLASS_NAMES = [
    "crane", "spreader", "spreader_corner", "hook",
    "container", "container_corner", "AGV", "lane_marker"
]

VALID_EXT = [".jpg", ".jpeg", ".png"]

# =========================
# DIR SETUP
# =========================
def create_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)

# =========================
# VISUALIZATION
# =========================
# =========================
# CLASS COLORS (BGR format for OpenCV)
# =========================
CLASS_COLORS = {
    0: (255, 0, 0),     # crane - blue
    1: (0, 255, 0),     # spreader - green
    2: (0, 255, 255),   # spreader_corner - yellow
    3: (255, 255, 0),   # hook - cyan
    4: (0, 0, 255),     # container - red
    5: (255, 0, 255),   # container_corner - magenta
    6: (128, 0, 255),   # AGV - purple
    7: (0, 165, 255),   # lane_marker - orange
}

# =========================
# VISUALIZATION
# =========================
def draw_boxes(img, boxes):
    img_copy = img.copy()
    h, w = img.shape[:2]

    for cls, x, y, bw, bh in boxes:
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        color = CLASS_COLORS.get(cls, (255, 255, 255))  # default white

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = CLASS_NAMES[cls]

        # Background for text (better visibility)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)

        # Put text
        cv2.putText(img_copy, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_copy

def save_debug(stage, name, img):
    path = os.path.join(DEBUG_DIR, f"{name}_{stage}.jpg")
    cv2.imwrite(path, img)

# =========================
# LABEL UTIL
# =========================
def read_labels(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append([int(cls), x, y, w, h])
    return boxes

def write_labels(label_path, boxes):
    with open(label_path, "w") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x} {y} {w} {h}\n")

# =========================
# WATERMARK REMOVAL + LABEL FIX
# =========================
def remove_watermark_and_adjust(img, boxes):
    h, w = img.shape[:2]

    crop_h = int(h * 0.92)
    crop_w = int(w * 0.92)

    new_boxes = []

    for cls, x, y, bw, bh in boxes:
        px = x * w
        py = y * h
        pw = bw * w
        ph = bh * h

        # keep only boxes inside cropped area
        if px < crop_w and py < crop_h:
            new_boxes.append([
                cls,
                px / crop_w,
                py / crop_h,
                pw / crop_w,
                ph / crop_h
            ])

    img = img[0:crop_h, 0:crop_w]

    return img, new_boxes

# =========================
# LOAD FILES
# =========================
def load_valid_pairs():
    images = os.listdir(INPUT_IMAGE_DIR)
    labels = os.listdir(INPUT_LABEL_DIR)

    image_map = {}
    for img in images:
        stem = Path(img).stem
        ext = Path(img).suffix.lower()
        if ext in VALID_EXT:
            image_map[stem] = img

    label_set = set([Path(f).stem for f in labels])
    common = list(set(image_map.keys()) & label_set)
    common.sort()

    return common, image_map

# =========================
# SPLIT
# =========================
def split_data(files):
    random.shuffle(files)
    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return (
        files[:train_end],
        files[train_end:val_end],
        files[val_end:]
    )

# =========================
# MAIN
# =========================
def process():
    create_dirs()
    files, image_map = load_valid_pairs()

    print(f"✅ Total valid pairs: {len(files)}")

    debug_samples = set(random.sample(files, min(DEBUG_SAMPLES, len(files))))

    train, val, test = split_data(files)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    counter = 0

    for split_name, split_files in zip(
        ["train", "val", "test"], [train, val, test]
    ):
        for fname in split_files:

            img_filename = image_map[fname]
            img_path = os.path.join(INPUT_IMAGE_DIR, img_filename)
            lbl_path = os.path.join(INPUT_LABEL_DIR, fname + ".txt")

            img = cv2.imread(img_path)
            boxes = read_labels(lbl_path)

            # DEBUG: original
            if DEBUG and fname in debug_samples:
                vis = draw_boxes(img, boxes)
                save_debug("original", fname, vis)

            # STEP: watermark removal + label adjust
            img, boxes = remove_watermark_and_adjust(img, boxes)

            if DEBUG and fname in debug_samples:
                vis = draw_boxes(img, boxes)
                save_debug("cropped", fname, vis)

            new_name = f"frame_{counter:06d}"

            out_img = f"{OUTPUT_DIR}/images/{split_name}/{new_name}.jpg"
            out_lbl = f"{OUTPUT_DIR}/labels/{split_name}/{new_name}.txt"

            cv2.imwrite(out_img, img)
            write_labels(out_lbl, boxes)

            counter += 1

    create_yaml()

# =========================
# YAML
# =========================
def create_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")

    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print("✅ data.yaml created")

# =========================
if __name__ == "__main__":
    process()