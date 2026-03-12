import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr
import torch
import torchreid
from datetime import datetime
import random
import os
import pickle
from scipy.spatial.distance import cosine
import re

# =========================
# CONFIG
# =========================

VIDEO_SOURCE = "video2.mp4" 
MODEL_PATH = "person_detector.pt"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DOOR_PT1 = (200, 150)  
DOOR_PT2 = (350, 550)  

ROI = {
    "x1": 20,
    "y1": 100,
    "x2": 700,
    "y2": 700
}

TIMESTAMP_ROI = (0, 0, 650, 80)

# Increased threshold safely because Normalization is now active
SIMILARITY_THRESHOLD = 0.30  
MAX_EMBEDDINGS_PER_ID = 15   # Larger flipbook to capture full 360-degree turning
COOLDOWN_FRAMES = 60 

OUTPUT_VIDEO = "attendance_output.mp4"
OUTPUT_EXCEL = "attendance.xlsx"
DB_FILE = "system_state.pkl" 

# =========================
# LOAD MODELS
# =========================

print("Loading YOLO...")
detector = YOLO(MODEL_PATH)

print("Loading OCR...")
reader = easyocr.Reader(['en'])

print("Loading ReID...")
reid_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
reid_model.eval()

# =========================
# DATABASE STRUCTURES
# =========================

employee_embeddings = {}
tracker_to_employee = {}
track_frame_count = {}  
tracker_last_seen = {} 
employee_id_counter = 1

track_history = {}
attendance_records = {}
employee_state = {}  
employee_colors = {}
last_event_frame = {} 

entry_count = 0
exit_count = 0
frame_counter = 0

# =========================
# PERSISTENCE FUNCTIONS
# =========================

def load_database():
    global employee_embeddings, employee_id_counter, attendance_records
    global employee_state, entry_count, exit_count, employee_colors

    if os.path.exists(DB_FILE):
        print(f"Loading previous memory from {DB_FILE}...")
        with open(DB_FILE, 'rb') as f:
            data = pickle.load(f)
            employee_embeddings = data.get('embeddings', {})
            employee_id_counter = data.get('id_counter', 1)
            attendance_records = data.get('records', {})
            employee_state = data.get('state', {})
            entry_count = data.get('entry_count', 0)
            exit_count = data.get('exit_count', 0)
            employee_colors = data.get('colors', {})
        print("Memory loaded successfully!")
    else:
        print("No previous memory found. Starting fresh.")

def save_database():
    print(f"Saving system memory to {DB_FILE}...")
    data = {
        'embeddings': employee_embeddings,
        'id_counter': employee_id_counter,
        'records': attendance_records,
        'state': employee_state,
        'entry_count': entry_count,
        'exit_count': exit_count,
        'colors': employee_colors
    }
    with open(DB_FILE, 'wb') as f:
        pickle.dump(data, f)

# =========================
# CORE FUNCTIONS
# =========================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

def get_position_state(px, py, pt1, pt2, current_state=None):
    x1, y1 = pt1
    x2, y2 = pt2
    d = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    
    MARGIN = 15000  
    
    if d > MARGIN:
        return "lobby"
    elif d < -MARGIN:
        return "office"
    else:
        if current_state is not None:
            return current_state
        else:
            return "lobby" if d > 0 else "office"

def get_color(emp_id):
    if emp_id not in employee_colors:
        employee_colors[emp_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    return employee_colors[emp_id]

def get_timestamp(frame):
    x1, y1, x2, y2 = TIMESTAMP_ROI
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = reader.readtext(enlarged)

    if len(result) > 0:
        full_text = " ".join([res[1] for res in result])
        if len(full_text) > 10:
            date_part = full_text[:10].replace('7', '/')
            full_text = date_part + full_text[10:]
        return full_text
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def parse_datetime(timestamp_str):
    time_match = re.search(r'\d{2}:\d{2}:\d{2}|\d{2}:\d{2}', timestamp_str)
    date_match = re.search(r'\d{2}/\d{2}/\d{4}', timestamp_str)
    now = datetime.now()
    date_str = date_match.group() if date_match else now.strftime("%d/%m/%Y")
    time_str = time_match.group() if time_match else now.strftime("%H:%M:%S")
    return date_str, time_str

def extract_embedding(person):
    img = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))
    
    # FIX 1: Convert to float and properly normalize using PyTorch standard ImageNet values
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # C, H, W
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = (img - mean) / std

    tensor = torch.from_numpy(np.expand_dims(img, axis=0))
    with torch.no_grad():
        return reid_model(tensor).numpy()[0]

def match_employee(embedding, visible_emp_ids):
    global employee_id_counter
    best_id, best_distance = None, 999

    for emp_id, emb_list in employee_embeddings.items():
        if emp_id in visible_emp_ids:
            continue
            
        for stored_emb in emb_list:
            dist = cosine(embedding, stored_emb)
            if dist < best_distance:
                best_distance = dist
                best_id = emp_id

    if best_distance < SIMILARITY_THRESHOLD:
        if best_id not in visible_emp_ids:
            if best_distance > 0.05:
                if len(employee_embeddings[best_id]) < MAX_EMBEDDINGS_PER_ID:
                    employee_embeddings[best_id].append(embedding)
                else:
                    # FIX 2: Rolling Memory (FIFO) - Drop the oldest angle to save the new one
                    employee_embeddings[best_id].pop(0)
                    employee_embeddings[best_id].append(embedding)
            return best_id

    employee_embeddings[employee_id_counter] = [embedding]
    employee_id_counter += 1
    return employee_id_counter - 1

def save_attendance():
    df = pd.DataFrame(list(attendance_records.values()))
    if df.empty:
        df = pd.DataFrame(columns=["Employee_ID", "Entry_Time", "Entry_Date", "Exit_Time", "Exit_Date"])
    df.to_excel(OUTPUT_EXCEL, index=False)

# =========================
# VIDEO SETUP & EXECUTION
# =========================

load_database()

cap = cv2.VideoCapture(VIDEO_SOURCE)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20, (FRAME_WIDTH, FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_counter += 1 
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    results = detector.track(frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        raw_boxes = results[0].boxes.xyxy.cpu().numpy()
        raw_track_ids = results[0].boxes.id.cpu().numpy()

        keep_indices = []
        for i in range(len(raw_boxes)):
            keep = True
            for j in keep_indices:
                if compute_iou(raw_boxes[i], raw_boxes[j]) > 0.5:
                    keep = False
                    break
            if keep:
                keep_indices.append(i)
                
        boxes = [raw_boxes[i] for i in keep_indices]
        track_ids = [raw_track_ids[i] for i in keep_indices]

        for tid in track_ids:
            tracker_last_seen[tid] = frame_counter

        visible_emp_ids = []
        for tid in track_ids:
            if tid in tracker_to_employee:
                eid = tracker_to_employee[tid]
                if eid in visible_emp_ids:
                    del tracker_to_employee[tid] 
                else:
                    visible_emp_ids.append(eid)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            cx = int((x1 + x2) / 2)
            foot_y = y2 

            if not (ROI["x1"] < cx < ROI["x2"] and ROI["y1"] < foot_y < ROI["y2"]):
                continue
                
            if track_id not in track_frame_count:
                track_frame_count[track_id] = 0
            track_frame_count[track_id] += 1

            if track_frame_count[track_id] < 5:
                continue
                
            pad = 15
            y1_pad = max(0, y1 - pad)
            y2_pad = min(FRAME_HEIGHT, y2 + pad)
            x1_pad = max(0, x1 - pad)
            x2_pad = min(FRAME_WIDTH, x2 + pad)
            
            person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if person_crop.size == 0:
                continue

            if track_id not in tracker_to_employee:
                embedding = extract_embedding(person_crop)
                emp_id = match_employee(embedding, visible_emp_ids)
                
                if emp_id in tracker_to_employee.values():
                    if tracker_to_employee.get(track_id) != emp_id:
                        emp_id = employee_id_counter
                        employee_embeddings[emp_id] = [embedding]
                        employee_id_counter += 1
                
                tracker_to_employee[track_id] = emp_id
                visible_emp_ids.append(emp_id) 
                
                if emp_id not in employee_state:
                    employee_state[emp_id] = get_position_state(cx, foot_y, DOOR_PT1, DOOR_PT2)
            else:
                emp_id = tracker_to_employee[track_id]
                
                # Check their appearance more frequently to build a good breadcrumb trail
                if track_frame_count[track_id] % 10 == 0:
                    new_embedding = extract_embedding(person_crop)
                    
                    # Find distance to their OWN best matching angle
                    best_dist_to_own = min([cosine(new_embedding, emb) for emb in employee_embeddings[emp_id]])
                    
                    # If it's a new angle, but still safely them, update the flipbook!
                    if 0.05 < best_dist_to_own < (SIMILARITY_THRESHOLD + 0.05):
                        if len(employee_embeddings[emp_id]) < MAX_EMBEDDINGS_PER_ID:
                            employee_embeddings[emp_id].append(new_embedding)
                        else:
                            employee_embeddings[emp_id].pop(0) # FIFO
                            employee_embeddings[emp_id].append(new_embedding)

            emp_id = tracker_to_employee.get(track_id)
            if emp_id is None: 
                continue

            emp_color = get_color(emp_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), emp_color, 2)
            cv2.putText(frame, f"EID:{emp_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emp_color, 2)

            if emp_id not in track_history:
                track_history[emp_id] = []

            track_history[emp_id].append((cx, foot_y))
            if len(track_history[emp_id]) > 30:
                track_history[emp_id].pop(0)

            if len(track_history[emp_id]) >= 2 and track_frame_count[track_id] > 15:
                prev_cx, prev_cy = track_history[emp_id][-2]
                curr_cx, curr_cy = track_history[emp_id][-1]

                curr_state = get_position_state(cx, foot_y, DOOR_PT1, DOOR_PT2, employee_state.get(emp_id))

                if employee_state.get(emp_id) == "lobby" and curr_state == "office":
                    if frame_counter - last_event_frame.get(emp_id, -999) > COOLDOWN_FRAMES:
                        timestamp = get_timestamp(frame)
                        date_str, time_str = parse_datetime(timestamp)
                        
                        if emp_id not in attendance_records:
                            attendance_records[emp_id] = {"Employee_ID": emp_id, "Entry_Time": time_str, "Entry_Date": date_str, "Exit_Time": "-", "Exit_Date": "-"}
                        else:
                            attendance_records[emp_id]["Entry_Time"] = time_str
                            attendance_records[emp_id]["Entry_Date"] = date_str
                            attendance_records[emp_id]["Exit_Time"] = "-" 
                            attendance_records[emp_id]["Exit_Date"] = "-"
                        
                        employee_state[emp_id] = "office"
                        entry_count += 1
                        last_event_frame[emp_id] = frame_counter 

                elif employee_state.get(emp_id) == "office" and curr_state == "lobby":
                    if frame_counter - last_event_frame.get(emp_id, -999) > COOLDOWN_FRAMES:
                        timestamp = get_timestamp(frame)
                        date_str, time_str = parse_datetime(timestamp)
                        
                        if emp_id not in attendance_records:
                            attendance_records[emp_id] = {"Employee_ID": emp_id, "Entry_Time": "-", "Entry_Date": "-", "Exit_Time": time_str, "Exit_Date": date_str}
                        else:
                            attendance_records[emp_id]["Exit_Time"] = time_str
                            attendance_records[emp_id]["Exit_Date"] = date_str
                        
                        employee_state[emp_id] = "lobby"
                        exit_count += 1
                        last_event_frame[emp_id] = frame_counter 

        for tid in list(tracker_to_employee.keys()):
            if frame_counter - tracker_last_seen.get(tid, frame_counter) > 60:
                tracker_to_employee.pop(tid, None)
                track_frame_count.pop(tid, None)

    # Draw UI
    cv2.rectangle(frame, (ROI["x1"], ROI["y1"]), (ROI["x2"], ROI["y2"]), (255, 0, 0), 2)
    cv2.line(frame, DOOR_PT1, DOOR_PT2, (0, 0, 255), 3)

    overlay = frame.copy()
    cv2.rectangle(overlay, (FRAME_WIDTH - 270, 20), (FRAME_WIDTH - 20, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    cv2.putText(frame, f"ENTRY: {entry_count}", (FRAME_WIDTH - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"EXIT: {exit_count}", (FRAME_WIDTH - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Attendance System", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP & SAVE
# =========================
save_database()
save_attendance()

cap.release()
out.release()
cv2.destroyAllWindows()