from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort

# =========================
# CONFIG
# =========================
MODEL_PATH = "./runs/detect/v2/train-colab/weights/best.pt"
VIDEO_PATH = "../data/raw/videos/video_01.mp4"

CONF_THRESHOLD = 0.5

ALIGN_X_THRESH = 20
ALIGN_Y_THRESH = 20

ATTACH_DIST_THRESH = 30
STABLE_FRAMES = 5

MAX_MISSING_FRAMES = 15

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

# =========================
# TRACKERS
# =========================
spreader_tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)
container_tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)

# =========================
# HELPERS
# =========================
def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def prepare_detections(result, target_class, names):
    dets = []
    for box in result.boxes:
        cls = int(box.cls[0])
        if names[cls] == target_class:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            dets.append([x1, y1, x2, y2, conf])
    return np.array(dets) if len(dets) > 0 else np.empty((0,5))

def pick_main_track(tracks):
    areas = []
    for t in tracks:
        x1, y1, x2, y2, _ = t
        areas.append((x2-x1)*(y2-y1))
    return tracks[np.argmax(areas)]

def pick_closest_container(spreader, containers):
    sx1, sy1, sx2, sy2, _ = spreader
    s_cx, s_cy = get_center((sx1, sy1, sx2, sy2))

    best = None
    best_score = float("inf")

    for c in containers:
        x1, y1, x2, y2, cid = c
        c_cx, c_cy = get_center((x1, y1, x2, y2))

        if c_cy < s_cy - 50:
            continue

        dx = abs(s_cx - c_cx)
        dy = abs(s_cy - c_cy)

        score = dx * 2 + dy

        print(f"      Candidate C_ID {int(cid)} → dx={dx}, dy={dy}, score={score}")

        if score < best_score:
            best_score = score
            best = c

    return best

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

prev_spreader_y = None
prev_container_y = None

stable_counter = 0
state = "INIT"
frame_id = 0

last_spreader = None
last_container = None

missing_counter = 0

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    print(f"\n================ FRAME {frame_id} ================")

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    # =========================
    # DETECTIONS
    # =========================
    spreader_dets = prepare_detections(results, "spreader", model.names)
    container_dets = prepare_detections(results, "container", model.names)

    print(f"Detections → Spreaders: {len(spreader_dets)}, Containers: {len(container_dets)}")

    # =========================
    # TRACKING
    # =========================
    spreader_tracks = spreader_tracker.update(spreader_dets)
    container_tracks = container_tracker.update(container_dets)

    print(f"Tracks → Spreaders: {len(spreader_tracks)}, Containers: {len(container_tracks)}")

    # =========================
    # SPREADER
    # =========================
    if len(spreader_tracks) > 0:
        spreader = pick_main_track(spreader_tracks)
        last_spreader = spreader
        print(f"Selected Spreader ID: {int(spreader[4])}")
    else:
        spreader = last_spreader
        print("⚠️ Using LAST spreader")

    # =========================
    # CONTAINER
    # =========================
    if spreader is not None and len(container_tracks) > 0:
        print("Evaluating container candidates...")
        container = pick_closest_container(spreader, container_tracks)

        if container is not None:
            last_container = container
            print(f"Selected Container ID: {int(container[4])}")
        else:
            print("⚠️ No valid container → fallback to ANY container")
            container = pick_main_track(container_tracks)
            last_container = container
    else:
        container = last_container
        print("⚠️ Using LAST container")

    # =========================
    # MISSING HANDLING
    # =========================
    if spreader is None or container is None:
        missing_counter += 1
    else:
        missing_counter = 0

    print(f"Missing Counter: {missing_counter}")

    if missing_counter > MAX_MISSING_FRAMES:
        print("❌ HARD FAIL: Missing objects too long")

        cv2.putText(frame, "Missing objects", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Debug", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # =========================
    # SAFE TO CONTINUE
    # =========================
    sx1, sy1, sx2, sy2, sid = spreader
    cx1, cy1, cx2, cy2, cid = container

    s_cx, s_cy = get_center((sx1, sy1, sx2, sy2))
    c_cx, c_cy = get_center((cx1, cy1, cx2, cy2))

    dx = s_cx - c_cx
    dy = s_cy - c_cy
    error = np.sqrt(dx**2 + dy**2)

    print(f"DX={dx}, DY={dy}, ERROR={error:.2f}")

    # =========================
    # VISUALS
    # =========================
    cv2.circle(frame, (s_cx, s_cy), 5, (0,255,0), -1)
    cv2.circle(frame, (c_cx, c_cy), 5, (0,0,255), -1)

    cv2.line(frame, (s_cx, s_cy), (c_cx, c_cy), (255,255,0), 2)

    cv2.putText(frame, f"S_ID: {int(sid)}", (s_cx, s_cy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.putText(frame, f"C_ID: {int(cid)}", (c_cx, c_cy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.putText(frame, f"DX: {dx}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"DY: {dy}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"Err: {int(error)}", (50, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Alignment Temporal Debug", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Finished")