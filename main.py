import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import psycopg2
from datetime import datetime, date as _date
from dotenv import load_dotenv
import os

# =========================
# DB CONNECTION
# =========================
load_dotenv()

# KeyError if .env missing = clear error
conn = psycopg2.connect(
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASS"],
    host=os.environ.get("DB_HOST", "localhost"),
    port=os.environ.get("DB_PORT", "5432")
)
cursor = conn.cursor()

# =========================
# CONFIG
# =========================
VIDEO_PATH = "CENTURY_DATA.mp4" # Or set to 0 for live webcam
MODEL_PATH = "yolov8s.pt"

DOOR_PT1 = (500, 100)
DOOR_PT2 = (800, 650)

FACE_THRESHOLD = 0.4        # Unchanged — anti-clone fix handles the symptom
SIMILARITY_THRESHOLD = 0.50 # Unchanged
COOLDOWN_FRAMES = 150       # Increased — fixes phantom exits

# =========================
# MODELS
# =========================
print("Loading Models...")
detector = YOLO(MODEL_PATH)

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(1280, 1280))

# =========================
# LOAD EMPLOYEES (RAM CACHE)
# =========================
cursor.execute("SELECT eid, name FROM employees")
rows = cursor.fetchall()
name_map = {eid: name for eid, name in rows}
print(f"Loaded {len(name_map)} registered employees from database.")

# 🔥 FIX 1: Unpack the massive byte block back into a 2D array
cursor.execute("SELECT eid, face_embedding FROM employees WHERE face_embedding IS NOT NULL")
face_db = {}
for eid, emb in cursor.fetchall():
    # .reshape(-1, 512) splits the giant byte chunk back into individual 512-number faces
    face_db[eid] = np.frombuffer(bytes(emb), dtype=np.float32).reshape(-1, 512)
    
print(f"Loaded {len(face_db)} employee face galleries into memory.")

current_date = _date.today()

# =========================
# MEMORY
# =========================
tracker_to_employee = {}
track_history = {}
raw_track_history = {} # Used for stability filter
last_event_frame = {}
daily_reid_db = {}

# Dynamic labels for guests/strangers
label_map = {}
label_counter = 0

entry_count = 0
exit_count = 0
frame_counter = 0

# DB CACHE
cached_attendance = []
db_needs_update = True

# =========================
# UTILS
# =========================
def side_of_line(x, y):
    x1, y1 = DOOR_PT1
    x2, y2 = DOOR_PT2
    return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

def full_cross(prev, curr):
    return side_of_line(*prev) * side_of_line(*curr) < 0

# 🔥 FIX 2: Check the live face against ALL 15 saved frames
def match_face(face_emb: np.ndarray | None) -> int | None:
    if face_emb is None:
        return None
        
    best_eid = None
    best_dist = 999.0
    
    # Check against EVERY person in the database
    for eid, db_embs in face_db.items():
        # Check against EVERY saved angle for that person
        for db_emb in db_embs: 
            dist = cosine(face_emb, db_emb)
            if dist < best_dist:
                best_dist = dist
                best_eid = eid
                
    if best_dist < FACE_THRESHOLD:
        name = name_map.get(best_eid, str(best_eid))
        print(f"[Face Match] {name} (EID {best_eid}) dist={best_dist:.3f}")
        return best_eid
        
    print(f"[Face Match] No match — best dist={best_dist:.3f} (threshold={FACE_THRESHOLD})")
    return None

def match_face_to_person(faces, x1, y1, x2, y2):
    for face in faces:
        # Ignore blurry/distant faces to prevent Identity Theft
        if float(face.det_score) < 0.60:
            continue
            
        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2

        if x1 < cx < x2 and y1 < cy < y2:
            return face.embedding
    return None

def insert_entry(eid, current_name):
    now = datetime.now()
    cursor.execute("""
        INSERT INTO attendance (eid, name, date, entry_time)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (eid, date)
        DO UPDATE SET entry_time = EXCLUDED.entry_time
    """, (eid, current_name, now.date(), now.time()))
    conn.commit()

def update_exit(eid):
    now = datetime.now()
    cursor.execute("""
        UPDATE attendance
        SET exit_time = %s
        WHERE eid = %s AND date = %s
    """, (now.time(), eid, now.date()))
    conn.commit()

def get_today_attendance():
    cursor.execute("""
        SELECT eid, name, entry_time, exit_time
        FROM attendance
        WHERE date = CURRENT_DATE
        ORDER BY entry_time DESC
    """)
    return cursor.fetchall()

# =========================
# VIDEO SETUP
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)

# =========================
# MAIN LOOP
# =========================
while True:
    # =========================
    # DAILY RESET
    # =========================
    if _date.today() != current_date:
        current_date = _date.today()
        daily_reid_db.clear()
        tracker_to_employee.clear()
        last_event_frame.clear()
        track_history.clear()
        raw_track_history.clear()
        label_map.clear()
        label_counter = 0
        entry_count = 0
        exit_count = 0
        print(f"[Daily Reset] New day detected — all tracking state cleared for {current_date}")

    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (1280, 720))

    results = detector.track(frame, persist=True, conf=0.3, classes=[0], verbose=False)
    faces = face_app.get(frame)

    # DEBUG FACE BOXES
    for f in faces:
        fx1, fy1, fx2, fy2 = map(int, f.bbox)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

    if results[0].boxes.id is not None:
 
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        # ==========================================
        # ANTI-CLONING FIX: FIND WHO IS ON SCREEN
        # ==========================================
        visible_eids_this_frame = set()
        for tid in ids:
            if tid in tracker_to_employee:
                visible_eids_this_frame.add(tracker_to_employee[tid])

        for box, tid in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = y2

            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                continue

            # =========================
            # STABILITY FILTER
            # =========================
            raw_track_history.setdefault(tid, []).append((cx, cy))
            if len(raw_track_history[tid]) < 5:
                continue

            # =========================
            # IDENTIFICATION LOGIC (FACE ONLY)
            # =========================
            # 1. Is this person new, or currently an unidentified "Guest" (EID >= 1000)?
            is_guest = (tid in tracker_to_employee and tracker_to_employee[tid] >= 1000)

            # 2. Keep hunting for a face EVERY frame if they are a Guest or Brand New
            if tid not in tracker_to_employee or is_guest:
                assigned_eid = None
                
                # Try to find a face match
                face_emb = match_face_to_person(faces, x1, y1, x2, y2)
                if face_emb is not None:
                    assigned_eid = match_face(face_emb)
                    
                    # Anti-clone check: Reject if this face matched someone already on screen
                    if assigned_eid is not None and assigned_eid in visible_eids_this_frame:
                        assigned_eid = None

                # 3. UPGRADE: If we found a valid employee face, assign their real ID!
                if assigned_eid is not None:
                    tracker_to_employee[tid] = assigned_eid
                    visible_eids_this_frame.add(assigned_eid)
                    last_event_frame.setdefault(assigned_eid, -999)

            # 4. DEFAULT: If they are totally new and we STILL haven't seen a face, make them a Guest
            if tid not in tracker_to_employee:
                new_id = 1000 + int(tid)
                tracker_to_employee[tid] = new_id
                visible_eids_this_frame.add(new_id)
                last_event_frame.setdefault(new_id, -999)

            # =========================
            # DISPLAY NAME RESOLUTION
            # =========================
            if tid not in tracker_to_employee:
                eid = -1
                name = "Unknown"
            else:
                eid = tracker_to_employee[tid]
                
                if eid in name_map:
                    name = name_map[eid]
                else:
                    if eid not in label_map:
                        if label_counter < 26:
                            # Name strangers "Guest" instead of "Mr"
                            label_map[eid] = f"Guest {chr(65 + label_counter)}" 
                        else:
                            label_map[eid] = f"Guest {label_counter + 1}"
                        label_counter += 1
                    name = label_map[eid]

            # Draw Bounding Box
            color = (0, 255, 0) if eid != -1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if eid == -1:
                continue

            # =========================
            # ENTRY / EXIT LOGIC
            # =========================
            # Track movement by YOLO 'tid', NOT 'eid'!
            track_history.setdefault(tid, []).append((cx, cy))
            if len(track_history[tid]) > 20:
                track_history[tid].pop(0)

            if len(track_history[tid]) >= 2:
                prev = track_history[tid][-2]
                curr = track_history[tid][-1]

                if full_cross(prev, curr):
                    if frame_counter - last_event_frame[eid] > COOLDOWN_FRAMES:
                        prev_side = side_of_line(*prev)
                        curr_side = side_of_line(*curr)

                        if prev_side > 0 and curr_side < 0:
                            entry_count += 1
                            insert_entry(eid, name)  
                            db_needs_update = True

                        elif prev_side < 0 and curr_side > 0:
                            exit_count += 1
                            update_exit(eid)
                            db_needs_update = True

                        last_event_frame[eid] = frame_counter

    # =========================
    # HUD BACKGROUND & LINE
    # =========================
    overlay = frame.copy()
    cv2.rectangle(overlay, (950, 20), (1270, 130), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 150), (500, 400), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    
    cv2.line(frame, DOOR_PT1, DOOR_PT2, (0, 0, 255), 3)

    # =========================
    # HUD TEXT
    # =========================
    cv2.putText(frame, f"ENTRY: {entry_count}", (970, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"EXIT: {exit_count}", (970, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # =========================
    # INSTANT DB CACHE
    # =========================
    if db_needs_update:
        cached_attendance = get_today_attendance()
        db_needs_update = False

    y_offset = 180
    for eid_db, name_db, entry_time, exit_time in cached_attendance:
        en_str = entry_time.strftime("%H:%M:%S") if entry_time else "--"
        ex_str = exit_time.strftime("%H:%M:%S") if exit_time else "--"
        text = f"{name_db} ({eid_db}) IN: {en_str} OUT: {ex_str}"

        cv2.putText(frame, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(25) == 27: # Press ESC to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()