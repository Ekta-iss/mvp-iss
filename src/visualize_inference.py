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

CONF_THRESHOLD = 0.3
LOCK_DIST_PX = 80       
UNLOCK_DIST_PX = 120    
REID_PROXIMITY_PX = 60  
MAX_LOST_AGE = 50       
MIN_STABILITY_FRAMES = 3 

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

def get_center(box):
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

# State Management (Only for Containers)
target_id = None
container_lost_registry = {}    
container_active_tracks = {}    

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

logger.info("--- SYSTEM START: Container-Only Tracking Mode ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_id += 1

    # Run detection/tracking
    results = model.track(frame, persist=True, tracker="botsort.yaml", conf=CONF_THRESHOLD, verbose=False)
    
    current_containers = []
    other_detections = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Check if tracking IDs are available (only relevant for containers now)
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)

        for box, yid, cls in zip(boxes, ids, clss):
            c_name = CLASS_NAMES[cls]
            det_data = {"yid": yid, "box": box, "center": get_center(box), "class": c_name}
            
            if c_name == "container":
                current_containers.append(det_data)
            else:
                other_detections.append(det_data)

    # --- CONTAINER TRACKING LOGIC ---
    new_container_tracks = {}
    unmatched_containers = []

    # 1. Update Existing Container Tracks
    for det in current_containers:
        yid = det["yid"]
        if yid in container_active_tracks:
            container_active_tracks[yid].update({
                "center": det["center"], "box": det["box"], 
                "hits": container_active_tracks[yid].get("hits", 0) + 1
            })
            new_container_tracks[yid] = container_active_tracks[yid]
        else:
            unmatched_containers.append(det)

    # 2. Re-ID for Lost Containers
    if unmatched_containers and container_lost_registry:
        lost_ids = list(container_lost_registry.keys())
        cost_matrix = np.zeros((len(lost_ids), len(unmatched_containers)))

        for i, l_id in enumerate(lost_ids):
            for j, det in enumerate(unmatched_containers):
                cost_matrix[i, j] = np.linalg.norm(np.array(det["center"]) - np.array(container_lost_registry[l_id]["last_pos"]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_idxs = set()

        for l_idx, d_idx in zip(row_ind, col_ind):
            if cost_matrix[l_idx, d_idx] < REID_PROXIMITY_PX:
                old_id = lost_ids[l_idx]
                det = unmatched_containers[d_idx]
                new_container_tracks[old_id] = {
                    "center": det["center"], "box": det["box"], 
                    "class": "container", "hits": MIN_STABILITY_FRAMES
                }
                logger.info(f"F:{frame_id} | RE-ID | Recovered Container {old_id}")
                container_lost_registry.pop(old_id)
                assigned_idxs.add(d_idx)
        unmatched_containers = [d for i, d in enumerate(unmatched_containers) if i not in assigned_idxs]

    # 3. New Container IDs
    for det in unmatched_containers:
        new_container_tracks[det["yid"]] = {"center": det["center"], "box": det["box"], "class": "container", "hits": 1}
        
    # 4. Handle Vanishing Containers
    for tid in list(container_active_tracks.keys()):
        if tid not in new_container_tracks:
            container_lost_registry[tid] = {"last_pos": container_active_tracks[tid]["center"], "frame": frame_id}

    container_lost_registry = {k: v for k, v in container_lost_registry.items() if (frame_id - v["frame"]) < MAX_LOST_AGE}
    container_active_tracks = new_container_tracks

    # --- VISUALIZATION ---
    
    # A. Draw Static Detections (Crane, Spreader, Lane Marker)
    spreader_center = None
    for det in other_detections:
        box, c_name = det["box"], det["class"]
        color = CLASS_COLORS.get(c_name, (255, 255, 255))
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame, f"{c_name}", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if c_name == "spreader":
            spreader_center = det["center"]

    # B. Draw Tracked Containers
    for tid, data in container_active_tracks.items():
        if data["hits"] < MIN_STABILITY_FRAMES: continue
        box = data["box"]
        color = CLASS_COLORS["container"]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame, f"ID:{tid} container", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # C. Sticky Lock (Spreader to Container)
    if spreader_center:
        if target_id is not None:
            if target_id in container_active_tracks:
                dist = np.linalg.norm(np.array(spreader_center) - np.array(container_active_tracks[target_id]["center"]))
                if dist > UNLOCK_DIST_PX:
                    target_id = None
            else: target_id = None

        if target_id is None:
            for tid, data in container_active_tracks.items():
                dist = np.linalg.norm(np.array(spreader_center) - np.array(data["center"]))
                if dist < LOCK_DIST_PX:
                    target_id = tid
                    break

        if target_id and target_id in container_active_tracks:
            cv2.line(frame, spreader_center, container_active_tracks[target_id]["center"], (0, 0, 255), 2)

    cv2.imshow("SmartVision Container Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()