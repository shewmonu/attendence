import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import psycopg2
from datetime import datetime, date as _date
import easyocr
import re
from dotenv import load_dotenv
import os

# =========================
# DB CONNECTION
# =========================
load_dotenv()

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
VIDEO_PATH = "CENTURY_DATA.mp4" 
MODEL_PATH = "yolov8s.pt"
OUTPUT_VIDEO = "output_attendance.mp4"

# Single Entry/Exit Line
DOOR_PT1 = (500, 100) 
DOOR_PT2 = (800, 650) 

FACE_THRESHOLD = 0.4        
SIMILARITY_THRESHOLD = 0.50 
COOLDOWN_FRAMES = 150       

# =========================
# MODELS
# =========================
print("Loading Models...")
detector = YOLO(MODEL_PATH)

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(1280, 1280))

print("Loading EasyOCR Model...")
ocr_reader = easyocr.Reader(['en'], gpu=True) 

# =========================
# LOAD EMPLOYEES
# =========================
cursor.execute("SELECT eid, name FROM employees")
rows = cursor.fetchall()
name_map = {eid: name for eid, name in rows}
print(f"Loaded {len(name_map)} registered employees from database.")

cursor.execute("SELECT eid, face_embedding FROM employees WHERE face_embedding IS NOT NULL")
face_db = {}
for eid, emb in cursor.fetchall():
    face_db[eid] = np.frombuffer(bytes(emb), dtype=np.float32).reshape(-1, 512)
    
print(f"Loaded {len(face_db)} employee face galleries into memory.")

current_date = _date.today()
current_time = datetime.min.time()

# =========================
# MEMORY
# =========================
tracker_to_employee = {}
track_history = {}
last_event_frame = {}

entry_count = 0
exit_count = 0
frame_counter = 0

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

def match_face(face_emb: np.ndarray | None) -> int | None:
    if face_emb is None: return None
    best_eid = None
    best_dist = 999.0
    for eid, db_embs in face_db.items():
        for db_emb in db_embs: 
            dist = cosine(face_emb, db_emb)
            if dist < best_dist:
                best_dist = dist
                best_eid = eid
    if best_dist < FACE_THRESHOLD:
        name = name_map.get(best_eid, str(best_eid))
        print(f"[Face Match] {name} (EID {best_eid}) dist={best_dist:.3f}")
        return best_eid
    return None

def match_face_to_person(faces, x1, y1, x2, y2):
    for face in faces:
        if float(face.det_score) < 0.60: continue
        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        if x1 < cx < x2 and y1 < cy < y2:
            return face.embedding
    return None

def get_time_from_footage(frame, fallback_date, fallback_time):
    try:
        crop = frame[30:150, 40:850]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        results = ocr_reader.readtext(gray, detail=0)
        text = " ".join(results)
        
        match = re.search(r"(\d{2})\D+(\d{2})\D+(\d{4}).*?([0-2]\d).?([0-5]\d).?([0-5]\d)", text)
        if match:
            d, m, y, h, minute, s = match.groups()
            footage_date = datetime.strptime(f"{d}/{m}/{y}", "%d/%m/%Y").date()
            footage_time = datetime.strptime(f"{h}:{minute}:{s}", "%H:%M:%S").time()
            return footage_date, footage_time
    except Exception as e:
        pass
    return fallback_date, fallback_time 

def insert_entry(eid, current_name, event_date, event_time):
    cursor.execute("""
        INSERT INTO attendance (eid, name, date, entry_time)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (eid, date)
        DO UPDATE SET entry_time = EXCLUDED.entry_time
    """, (eid, current_name, event_date, event_time))
    conn.commit()

def update_exit(eid, event_date, event_time):
    cursor.execute("""
        UPDATE attendance
        SET exit_time = %s
        WHERE eid = %s AND date = %s
    """, (event_time, eid, event_date))
    conn.commit()

def get_today_attendance(target_date):
    cursor.execute("""
        SELECT eid, name, entry_time, exit_time
        FROM attendance
        WHERE date = %s
        ORDER BY entry_time DESC
    """, (target_date,))
    return cursor.fetchall()

# =========================
# VIDEO SETUP
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 25.0

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (1280, 720))
cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_counter += 1
    frame = cv2.resize(frame, (1280, 720))

    if frame_counter % 30 == 0:
        polled_date, polled_time = get_time_from_footage(frame, current_date, current_time)
        if polled_date != current_date:
            current_date = polled_date
            entry_count, exit_count = 0, 0
            tracker_to_employee.clear()
            track_history.clear()
            print(f"[System] Date changed to {current_date}")
        current_time = polled_time

    results = detector.track(frame, persist=True, conf=0.3, classes=[0], verbose=False)
    faces = face_app.get(frame)

    for f in faces:
        fx1, fy1, fx2, fy2 = map(int, f.bbox)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        visible_eids_this_frame = set()
        for tid in ids:
            if tid in tracker_to_employee:
                visible_eids_this_frame.add(tracker_to_employee[tid])

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Tracking center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            is_guest = (tid in tracker_to_employee and tracker_to_employee[tid] >= 1000)

            if tid not in tracker_to_employee or is_guest:
                assigned_eid = None
                face_emb = match_face_to_person(faces, x1, y1, x2, y2)
                if face_emb is not None:
                    assigned_eid = match_face(face_emb)
                    if assigned_eid is not None and assigned_eid in visible_eids_this_frame:
                        assigned_eid = None

                if assigned_eid is not None:
                    tracker_to_employee[tid] = assigned_eid
                    visible_eids_this_frame.add(assigned_eid)
                    last_event_frame.setdefault(assigned_eid, -999)

            if tid not in tracker_to_employee:
                new_id = 1000 + int(tid)
                tracker_to_employee[tid] = new_id
                visible_eids_this_frame.add(new_id)
                last_event_frame.setdefault(new_id, -999)

            if tid not in tracker_to_employee:
                eid = -1
                name = "Unknown"
            else:
                eid = tracker_to_employee[tid]
                if eid in name_map:
                    name = name_map[eid]
                else:
                    name = "Unknown"

            color = (0, 255, 0) if eid in name_map else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # --- CORE TRACKING LOGIC ---
            track_history.setdefault(tid, []).append((cx, cy))
            if len(track_history[tid]) > 20: track_history[tid].pop(0)

            if len(track_history[tid]) >= 10:
                # 7-Frame Momentum to fix jitters
                prev, curr = track_history[tid][-7], track_history[tid][-1]
                
                if frame_counter > 60:
                    if full_cross(prev, curr):
                        if frame_counter - last_event_frame.get(eid, -999) > COOLDOWN_FRAMES:
                            p_side = side_of_line(*prev)
                            c_side = side_of_line(*curr)
                            
                            # ENTRY
                            if p_side > 0 and c_side < 0:
                                print(f"[{name}] entered at {current_time}")
                                entry_count += 1
                                
                                # 🔥 Prevents Unknowns from cluttering DB
                                if name != "Unknown":
                                    insert_entry(eid, name, current_date, current_time)
                                    db_needs_update = True
                                    
                                last_event_frame[eid] = frame_counter

                            # EXIT
                            elif p_side < 0 and c_side > 0:
                                print(f"[{name}] exited at {current_time}")
                                exit_count += 1
                                
                                # 🔥 Prevents Unknowns from cluttering DB
                                if name != "Unknown":
                                    update_exit(eid, current_date, current_time)
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

    cv2.putText(frame, f"ENTRY: {entry_count}", (970, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"EXIT: {exit_count}", (970, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if db_needs_update:
        cached_attendance = get_today_attendance(current_date)
        db_needs_update = False

    y_offset = 180
    for eid_db, name_db, entry_time, exit_time in cached_attendance:
        en_str = entry_time.strftime("%H:%M") if entry_time else "--"
        ex_str = exit_time.strftime("%H:%M") if exit_time else "--"
        
        text = f"{name_db} IN: {en_str} OUT: {ex_str}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    out_writer.write(frame)
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(25) == 27: 
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()