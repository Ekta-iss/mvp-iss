import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
TARGET_WORK_ORDER = 33

MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"
VIDEO_PATH = "../data/raw/videos/video_01.mp4"

OUTPUT_DIR = "./runs/inference_outputs"
OUTPUT_VIDEO = "smartvision_inference.mp4"

# Visual Settings
CONF_THRESHOLD = 0.25
LINE_THICKNESS = 3
FONT_SCALE = 0.6

# =========================
# CLASS COLORS (BGR)
# =========================
CLASS_COLORS = {
    "crane": (255, 0, 0),       # Blue
    "spreader": (0, 255, 255),  # Yellow
    "container": (0, 255, 0),   # Green
    "target": (255, 120, 0),    # Orange/Azure
    "warning": (0, 0, 255)      # Red
}

# =========================
# CREATE OUTPUT DIR
# =========================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

output_video_path = str(Path(OUTPUT_DIR) / OUTPUT_VIDEO)

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# LOAD VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {VIDEO_PATH}")

# =========================
# VIDEO PROPERTIES
# =========================
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Video Loaded")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")

# =========================
# VIDEO WRITER
# =========================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (width, height)
)

print(f"💾 Saving inference video to:")
print(output_video_path)

# =========================
# TRACKING STATE
# =========================
inventory_seen = set()
frame_count = 0

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    h, w = frame.shape[:2]

    # =========================
    # UI CLEANUP
    # =========================
    cv2.rectangle(frame, (w - 180, h - 45), (w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 60), (25, 25, 25), -1)

    # =========================
    # YOLO TRACKING
    # =========================
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=CONF_THRESHOLD,
        verbose=False
    )

    spreader_center = None
    hovered_id = None

    # =========================
    # PROCESS DETECTIONS
    # =========================
    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        ids = (
            results[0].boxes.id.cpu().numpy().astype(int)
            if results[0].boxes.id is not None
            else [None] * len(boxes)
        )

        # =========================
        # FIND SPREADER CENTER
        # =========================
        for box, cls in zip(boxes, clss):

            if model.names[cls] == "spreader":

                spreader_center = (
                    int((box[0] + box[2]) / 2),
                    int((box[1] + box[3]) / 2)
                )

        # =========================
        # RENDER OBJECTS
        # =========================
        for box, yid, cls in zip(boxes, ids, clss):

            name = model.names[cls]

            b = box.astype(int)

            color = CLASS_COLORS.get(name, (255, 255, 255))

            # =========================
            # CONTAINER LOGIC
            # =========================
            if name == "container":

                inventory_seen.add(yid)

                label = f"CONT ID:{yid}" if yid is not None else "CONT"

                # Target container
                if yid == TARGET_WORK_ORDER:
                    color = CLASS_COLORS["target"]

                # Hover logic
                if spreader_center:

                    cnt_center = (
                        int((b[0] + b[2]) / 2),
                        int((b[1] + b[3]) / 2)
                    )

                    distance = np.linalg.norm(
                        np.array(spreader_center) - np.array(cnt_center)
                    )

                    if distance < 85:

                        hovered_id = yid

                        if yid != TARGET_WORK_ORDER:
                            color = CLASS_COLORS["warning"]

            else:
                label = name.upper()

            # =========================
            # DRAW BOX
            # =========================
            cv2.rectangle(
                frame,
                (b[0], b[1]),
                (b[2], b[3]),
                color,
                LINE_THICKNESS
            )

            # =========================
            # LABEL BACKGROUND
            # =========================
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                2
            )

            cv2.rectangle(
                frame,
                (b[0], b[1] - th - 10),
                (b[0] + tw, b[1]),
                color,
                -1
            )

            # =========================
            # LABEL TEXT
            # =========================
            cv2.putText(
                frame,
                label,
                (b[0], b[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                2
            )

    # =========================
    # DASHBOARD
    # =========================
    cv2.putText(
        frame,
        f"JOB ORDER: PICK UP #{TARGET_WORK_ORDER}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"INVENTORY COUNT: {len(inventory_seen)}",
        (w - 240, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # =========================
    # ALERT MESSAGE
    # =========================
    if hovered_id:

        if hovered_id == TARGET_WORK_ORDER:
            msg = "TARGET MATCHED"
            m_clr = CLASS_COLORS["container"]

        else:
            msg = f"WRONG CONTAINER (ID:{hovered_id})"
            m_clr = CLASS_COLORS["warning"]

        cv2.putText(
            frame,
            msg,
            (int(w / 2) - 150, h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            m_clr,
            3
        )

    # =========================
    # SAVE FRAME TO VIDEO
    # =========================
    video_writer.write(frame)

    # =========================
    # SHOW LIVE WINDOW
    # =========================
    cv2.imshow("SmartVision Logistics MVP", frame)

    # =========================
    # EXIT KEY
    # =========================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

# =========================
# CLEANUP
# =========================
cap.release()
video_writer.release()

cv2.destroyAllWindows()

print("\n✅ Inference completed")
print(f"🎥 Saved output video:")
print(output_video_path)