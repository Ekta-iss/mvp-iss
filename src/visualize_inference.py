import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"
VIDEO_PATH = "../data/raw/videos/video_01.mp4"

CONF_THRESHOLD = 0.7

# Class names (update if needed)
CLASS_NAMES = {
    0: "crane",
    1: "spreader",
    2: "container",
    3: "lane_marker"
}

# Colors for each class (BGR)
CLASS_COLORS = {
    0: (255, 0, 0),      # crane - blue
    1: (0, 255, 255),    # spreader - yellow
    2: (0, 255, 0),      # container - green
    3: (255, 0, 255)     # lane_marker - pink
}

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# LOAD VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # INFERENCE
    # =========================
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = CLASS_NAMES.get(cls_id, str(cls_id))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()