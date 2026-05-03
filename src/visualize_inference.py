import cv2
import numpy as np
import logging
import os
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"
VIDEO_PATH = "../data/raw/videos/video_01.mp4"
LOG_FILE = "crane_debug.log"

# Detection Hyperparameters
CONF_THRESHOLD = 0.15   
NMS_IOU_THRESHOLD = 0.4  
IOU_REID_THRESHOLD = 0.3 

# Geometry Tuning - Adjusted based on your logs (Ratio ~4.5)
EXPECTED_ASPECT_RATIO = 4.5  
RATIO_TOLERANCE = 1.8        # Allows range of 2.7 to 6.3

# Tracking Rules
LOCK_DIST_PX = 80       
UNLOCK_DIST_PX = 120    
MAX_LOST_AGE = 50       
MIN_STABILITY_FRAMES = 1 

# Visual Settings
LINE_THICKNESS = 3
FONT_SCALE = 0.8

# =========================
# INITIALIZATION & LOGGING
# =========================
if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("CRANE_APP")

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names
CLASS_COLORS = {
    "crane": (255, 0, 0), "spreader": (0, 255, 255), 
    "container": (0, 255, 0), "lane_marker": (255, 0, 255)
}

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(b1_area + b2_area - inter_area + 1e-6)

def get_center(box):
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

# State Management
target_id = None
container_lost_registry = {}    
container_active_tracks = {}    

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_id += 1

    results = model.track(
        frame, 
        persist=True, 
        tracker="botsort.yaml", 
        conf=CONF_THRESHOLD, 
        iou=NMS_IOU_THRESHOLD,
        verbose=False
    )
    
    current_containers = []
    other_detections = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)

        for box, yid, cls in zip(boxes, ids, clss):
            c_name = CLASS_NAMES[cls]
            
            if c_name == "container":
                w, h = (box[2] - box[0]), (box[3] - box[1])
                ratio = w / h if h != 0 else 0
                
                # Apply Geometry Filter based on your log data
                if abs(ratio - EXPECTED_ASPECT_RATIO) < RATIO_TOLERANCE:
                    current_containers.append({"yid": yid, "box": box, "center": get_center(box)})
            else:
                other_detections.append({"box": box, "class": c_name, "center": get_center(box)})

    # --- TRACKING LOGIC ---
    new_container_tracks = {}
    unmatched_containers = []

    for det in current_containers:
        yid = det["yid"]
        if yid is not None and yid in container_active_tracks:
            container_active_tracks[yid].update({"center": det["center"], "box": det["box"], "hits": container_active_tracks[yid].get("hits", 0) + 1})
            new_container_tracks[yid] = container_active_tracks[yid]
        else:
            unmatched_containers.append(det)

    # Re-ID logic
    if unmatched_containers and container_lost_registry:
        lost_ids = list(container_lost_registry.keys())
        cost_matrix = np.zeros((len(lost_ids), len(unmatched_containers)))
        for i, l_id in enumerate(lost_ids):
            for j, det in enumerate(unmatched_containers):
                cost_matrix[i, j] = 1.0 - calculate_iou(container_lost_registry[l_id]["last_box"], det["box"])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1.0 - IOU_REID_THRESHOLD):
                old_id = lost_ids[r]
                new_container_tracks[old_id] = {"center": unmatched_containers[c]["center"], "box": unmatched_containers[c]["box"], "hits": MIN_STABILITY_FRAMES}
                container_lost_registry.pop(old_id)

    for det in unmatched_containers:
        if det["yid"] is not None and det["yid"] not in new_container_tracks:
            new_container_tracks[det["yid"]] = {"center": det["center"], "box": det["box"], "hits": 1}

    for tid in list(container_active_tracks.keys()):
        if tid not in new_container_tracks:
            container_lost_registry[tid] = {"last_box": container_active_tracks[tid]["box"], "frame": frame_id}
    
    container_active_tracks = new_container_tracks
    container_lost_registry = {k: v for k, v in container_lost_registry.items() if (frame_id - v["frame"]) < MAX_LOST_AGE}

    # --- VISUALIZATION ---
    spreader_center = None
    
    # Draw non-container detections (Detection only, no Re-ID)
    for det in other_detections:
        b = det["box"].astype(int)
        color = CLASS_COLORS.get(det["class"], (255, 255, 255))
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, LINE_THICKNESS)
        cv2.putText(frame, det["class"], (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 2)
        if det["class"] == "spreader": spreader_center = det["center"]

    # Draw tracked containers with high-contrast labels
    for tid, data in container_active_tracks.items():
        if data["hits"] >= MIN_STABILITY_FRAMES:
            b = data["box"].astype(int)
            # Thick box
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), LINE_THICKNESS)
            
            # Label background for readability
            label = f"CONT ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
            cv2.rectangle(frame, (b[0], b[1] - th - 10), (b[0] + tw, b[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (b[0], b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 2)

    # Locking Visuals
    if spreader_center and target_id and target_id in container_active_tracks:
        cv2.line(frame, spreader_center, container_active_tracks[target_id]["center"], (0, 0, 255), 3)

    # Standard Lock Logic
    if spreader_center and target_id is None:
        for tid, data in container_active_tracks.items():
            if np.linalg.norm(np.array(spreader_center) - np.array(data["center"])) < LOCK_DIST_PX:
                target_id = tid
                logger.info(f"LOCKED Spreader to ID {tid}")
                break
    elif target_id and target_id in container_active_tracks:
        if np.linalg.norm(np.array(spreader_center) - np.array(container_active_tracks[target_id]["center"])) > UNLOCK_DIST_PX:
            target_id = None

    cv2.imshow("SmartVision MVP", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()