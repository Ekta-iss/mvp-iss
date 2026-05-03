import cv2
import logging
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"
VIDEO_PATH = "../data/raw/videos/video_01.mp4"

CONF_THRESHOLD = 0.3

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("CRANE_TRACK")

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# AUTO-GENERATE COLORS FOR ALL CLASSES
# =========================
import random
random.seed(42)

CLASS_NAMES = model.names  # IMPORTANT (truth source)

CLASS_COLORS = {
    name: tuple(random.randint(50, 255) for _ in range(3))
    for name in CLASS_NAMES.values()
}

# Force important consistency (optional override for clarity)
CLASS_COLORS.update({
    "crane": (255, 0, 0),
    "spreader": (0, 255, 255),
    "container": (0, 255, 0),
    "lane_marker": (255, 0, 255)
})

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0
active_tracks = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logger.info("Video ended.")
        break

    frame_id += 1
    logger.info(f"Frame {frame_id}")

    # =========================
    # TRACKING (ByteTrack)
    # =========================
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESHOLD,
        tracker="bytetrack.yaml"
    )

    r = results[0]

    if r.boxes is None:
        logger.warning(f"Frame {frame_id}: no detections")
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()

    track_ids = r.boxes.id
    if track_ids is not None:
        track_ids = track_ids.cpu().numpy().astype(int)
    else:
        track_ids = [-1] * len(boxes)

    current_tracks = set()

    # =========================
    # DRAW + DEBUG
    # =========================
    for box, cls, score, tid in zip(boxes, classes, scores, track_ids):

        x1, y1, x2, y2 = map(int, box)

        class_name = CLASS_NAMES[int(cls)]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        current_tracks.add(tid)

        # Track lifecycle logs
        if tid not in active_tracks:
            logger.info(f"NEW TRACK → ID:{tid} Class:{class_name}")

        logger.debug(
            f"Frame:{frame_id} ID:{tid} "
            f"Class:{class_name} ({cls}) "
            f"Conf:{score:.2f}"
        )

        label = f"{class_name} ID:{tid} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # =========================
    # LOST TRACKS
    # =========================
    for lost in active_tracks - current_tracks:
        logger.info(f"LOST TRACK → ID:{lost}")

    active_tracks = current_tracks

    # =========================
    # UI INFO
    # =========================
    cv2.putText(
        frame,
        f"Frame:{frame_id} Tracks:{len(active_tracks)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("ByteTrack - Crane System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()