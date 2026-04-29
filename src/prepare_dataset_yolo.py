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

OUTPUT_DIR = "../data/processed/final_yolo_dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

AUGMENT = True

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
# WATERMARK REMOVAL
# =========================

def remove_watermark(img):
    h, w = img.shape[:2]

    # Crop bottom-right region (adjust if needed)
    crop_h = int(h * 0.92)
    crop_w = int(w * 0.92)

    img = img[0:crop_h, 0:crop_w]

    return img

# =========================
# AUGMENTATION
# =========================

def augment_image(img):
    aug_images = []

    flip = cv2.flip(img, 1)
    aug_images.append(("flip", flip))

    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=25)
    aug_images.append(("bright", bright))

    return aug_images

def adjust_labels_for_flip(label_path):
    new_lines = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            x = 1 - x
            new_lines.append(f"{int(cls)} {x} {y} {w} {h}\n")
    return new_lines

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

            # STEP 1: remove watermark
            img = remove_watermark(img)

            # STEP 2: save normalized JPG
            new_name = f"frame_{counter:06d}"

            out_img = f"{OUTPUT_DIR}/images/{split_name}/{new_name}.jpg"
            out_lbl = f"{OUTPUT_DIR}/labels/{split_name}/{new_name}.txt"

            cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy(lbl_path, out_lbl)

            # STEP 3: augmentation ONLY for train
            if AUGMENT and split_name == "train":
                aug_list = augment_image(img)

                for aug_type, aug_img in aug_list:
                    aug_name = f"{new_name}_{aug_type}"

                    aug_img_path = f"{OUTPUT_DIR}/images/{split_name}/{aug_name}.jpg"
                    aug_lbl_path = f"{OUTPUT_DIR}/labels/{split_name}/{aug_name}.txt"

                    cv2.imwrite(aug_img_path, aug_img)

                    if aug_type == "flip":
                        new_lbl = adjust_labels_for_flip(lbl_path)
                        with open(aug_lbl_path, "w") as f:
                            f.writelines(new_lbl)
                    else:
                        shutil.copy(lbl_path, aug_lbl_path)

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