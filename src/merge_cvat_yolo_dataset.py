import shutil
from pathlib import Path

# =========================
# CONFIG (relative to src/)
# =========================
ROOT = Path("../data")

CVAT_DIR = ROOT / "annotations" / "cvat" / "video_01"
YOLO_DIR = ROOT / "annotations" / "yolo_labeled" / "video_01"

OUTPUT_DIR = ROOT / "processed" / "cvat_yolo_merged_dataset"

IMG_OUT = OUTPUT_DIR / "images"
LBL_OUT = OUTPUT_DIR / "labels"

REPORT_FILE = OUTPUT_DIR / "merge_report.txt"

# =========================
# CREATE OUTPUT STRUCTURE
# =========================
def setup_dirs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    IMG_OUT.mkdir(parents=True, exist_ok=True)
    LBL_OUT.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD FILES
# =========================
def get_files(folder):
    images = {p.name: p for p in (folder / "images").glob("*")}
    labels = {p.name: p for p in (folder / "labels").glob("*")}
    return images, labels

# =========================
# MERGE LOGIC
# =========================
def merge_datasets():
    cvat_imgs, cvat_lbls = get_files(CVAT_DIR)
    yolo_imgs, yolo_lbls = get_files(YOLO_DIR)

    report = []
    used = set()

    # STEP 1: Add CVAT data (priority dataset)
    for name, img_path in cvat_imgs.items():
        lbl_path = cvat_lbls.get(name.replace(".jpg", ".txt").replace(".png", ".txt"))

        shutil.copy(img_path, IMG_OUT / name)

        if lbl_path:
            shutil.copy(lbl_path, LBL_OUT / lbl_path.name)
        else:
            report.append(f"⚠️ Missing CVAT label for {name}")

        used.add(name)
        report.append(f"✔ CVAT used: {name}")

    # STEP 2: Add YOLO data (only if not in CVAT)
    for name, img_path in yolo_imgs.items():
        if name in used:
            continue

        lbl_path = yolo_lbls.get(name.replace(".jpg", ".txt").replace(".png", ".txt"))

        shutil.copy(img_path, IMG_OUT / name)

        if lbl_path:
            shutil.copy(lbl_path, LBL_OUT / lbl_path.name)
        else:
            report.append(f"⚠️ Missing YOLO label for {name}")

        report.append(f"✔ YOLO added: {name}")

    # STEP 3: Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n🎉 Merge completed!")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📝 Report: {REPORT_FILE}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("🚀 Merging CVAT + YOLO datasets...\n")

    setup_dirs()
    merge_datasets()

    print("\n✅ Done!")