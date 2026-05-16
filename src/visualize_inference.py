import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
TARGET_WORK_ORDER = 25  
MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"
VIDEO_PATH = "../data/raw/videos/video_01.mp4"

# Visual Settings
CONF_THRESHOLD = 0.25
LINE_THICKNESS = 3
FONT_SCALE = 0.6

# Class Colors (BGR)
CLASS_COLORS = {
    "crane": (255, 0, 0),      # Blue
    "spreader": (0, 255, 255), # Yellow
    "container": (0, 255, 0),  # Green
    "target": (255, 120, 0),   # Azure/Orange (For active Job Order)
    "warning": (0, 0, 255)     # Red (For incorrect picks)
}

# =========================
# INITIALIZATION
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
inventory_seen = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    # 1. UI Cleanup: Patch watermark and create top dashboard
    cv2.rectangle(frame, (w - 180, h - 45), (w, h), (0, 0, 0), -1) 
    cv2.rectangle(frame, (0, 0), (w, 60), (25, 25, 25), -1)

    results = model.track(frame, persist=True, tracker="botsort.yaml", conf=CONF_THRESHOLD, verbose=False)
    
    spreader_center = None
    hovered_id = None

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None]*len(boxes)

        # First pass: Identify Spreader position for proximity logic
        for box, cls in zip(boxes, clss):
            if model.names[cls] == "spreader":
                spreader_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

        # Second pass: Rendering and Logistics Logic
        for box, yid, cls in zip(boxes, ids, clss):
            name = model.names[cls]
            b = box.astype(int)
            color = CLASS_COLORS.get(name, (255, 255, 255))
            
            # --- Container Specific Logic ---
            if name == "container":
                inventory_seen.add(yid)
                label = f"CONT ID:{yid}" if yid is not None else "CONT"
                
                # Check if this is the target container
                if yid == TARGET_WORK_ORDER:
                    color = CLASS_COLORS["target"]
                
                # Check for "Hover" (Spreader proximity to Container)
                if spreader_center:
                    cnt_center = (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
                    if np.linalg.norm(np.array(spreader_center) - np.array(cnt_center)) < 85:
                        hovered_id = yid
                        if yid != TARGET_WORK_ORDER:
                            color = CLASS_COLORS["warning"]
            else:
                # Crane and Spreader: No IDs displayed
                label = name.upper()

            # --- Drawing ---
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, LINE_THICKNESS)
            
            # Label background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
            cv2.rectangle(frame, (b[0], b[1] - th - 10), (b[0] + tw, b[1]), color, -1)
            cv2.putText(frame, label, (b[0], b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 2)

    # --- DASHBOARD & ALERTS ---
    cv2.putText(frame, f"JOB ORDER: PICK UP #{TARGET_WORK_ORDER}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"INVENTORY COUNT: {len(inventory_seen)}", (w - 240, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if hovered_id:
        if hovered_id == TARGET_WORK_ORDER:
            msg, m_clr = "TARGET MATCHED", CLASS_COLORS["container"]
        else:
            msg, m_clr = f"WRONG CONTAINER (ID:{hovered_id})", CLASS_COLORS["warning"]
        
        cv2.putText(frame, msg, (int(w/2) - 150, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, m_clr, 3)

    cv2.imshow("SmartVision Logistics MVP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()