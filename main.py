import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr
import torch
import torchreid
from datetime import datetime
from scipy.spatial.distance import cosine
import random

# =========================
# CONFIG
# =========================

VIDEO_SOURCE = "video1_6.mp4"
MODEL_PATH = "person_detector.pt"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DOOR_LINE_X = 220 

ROI = {
    "x1": 20,
    "y1": 100,
    "x2": 700,
    "y2": 700
}

TIMESTAMP_ROI = (0, 0, 500, 80)
SIMILARITY_THRESHOLD = 0.15

OUTPUT_VIDEO = "attendance_output.mp4"
OUTPUT_EXCEL = "attendance.xlsx"

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
employee_id_counter = 1

track_history = {}

# Changed to a Dictionary to update rows instead of appending new ones
attendance_records = {}

# Tracks if ID is 'office' or 'lobby'
employee_state = {}  

# Dictionary to store unique colors for each EID
employee_colors = {}

entry_count = 0
exit_count = 0

# =========================
# FUNCTIONS
# =========================

def get_color(emp_id):
    if emp_id not in employee_colors:
        employee_colors[emp_id] = (
            random.randint(50, 255), 
            random.randint(50, 255), 
            random.randint(50, 255)
        )
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
    """Splits '05/03/2026 Thu 09:55:52' into Date and Time"""
    parts = timestamp_str.strip().split()
    if len(parts) >= 2:
        date_str = parts[0]
        time_str = parts[-1]
    else:
        date_str = timestamp_str
        time_str = "-"
    return date_str, time_str


def extract_embedding(person):
    img = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.0
    
    img = np.expand_dims(img, axis=0)
    tensor = torch.from_numpy(img).float()

    with torch.no_grad():
        emb = reid_model(tensor)

    return emb.numpy()[0]


def match_employee(embedding, visible_emp_ids):
    global employee_id_counter

    if len(employee_embeddings) == 0:
        employee_embeddings[employee_id_counter] = embedding
        employee_id_counter += 1
        return employee_id_counter - 1

    best_id = None
    best_distance = 999

    for emp_id, stored_emb in employee_embeddings.items():
        # SAFETY CHECK: If this ID is already in the current frame, skip it!
        if emp_id in visible_emp_ids:
            continue
            
        dist = cosine(embedding, stored_emb)
        if dist < best_distance:
            best_distance = dist
            best_id = emp_id

    if best_distance < SIMILARITY_THRESHOLD:
        return best_id

    # If no match is found, create a new ID
    employee_embeddings[employee_id_counter] = embedding
    employee_id_counter += 1
    return employee_id_counter - 1

def save_attendance():
    # Convert dictionary values to a list for pandas
    df = pd.DataFrame(list(attendance_records.values()))
    if df.empty:
        df = pd.DataFrame(columns=["Employee_ID", "Entry_Time", "Entry_Date", "Exit_Time", "Exit_Date"])
    df.to_excel(OUTPUT_EXCEL, index=False)


# =========================
# VIDEO SETUP
# =========================

cap = cv2.VideoCapture(VIDEO_SOURCE)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    20,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    results = detector.track(
        frame,
        persist=True,
        tracker="botsort.yaml"
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        # Gather all Employee IDs that are ALREADY visible in this specific frame
        visible_emp_ids = [
            tracker_to_employee[tid] 
            for tid in track_ids 
            if tid in tracker_to_employee
        ]

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if not (ROI["x1"] < cx < ROI["x2"] and ROI["y1"] < cy < ROI["y2"]):
                continue

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            if track_id not in tracker_to_employee:
                embedding = extract_embedding(person_crop)
                # Pass the visible IDs to prevent duplicates
                emp_id = match_employee(embedding, visible_emp_ids)
                tracker_to_employee[track_id] = emp_id
                
                # Add this newly assigned ID to the visible list so the NEXT person 
                # in this same frame doesn't accidentally steal it
                visible_emp_ids.append(emp_id)
                
                if emp_id not in employee_state:
                    if cx > DOOR_LINE_X:
                        employee_state[emp_id] = "lobby"
                    else:
                        employee_state[emp_id] = "office"


            emp_id = tracker_to_employee[track_id]
            emp_color = get_color(emp_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), emp_color, 2)
            cv2.putText(
                frame,
                f"EID:{emp_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                emp_color,
                2
            )

            if emp_id not in track_history:
                track_history[emp_id] = []

            track_history[emp_id].append((cx, cy))

            if len(track_history[emp_id]) > 30:
                track_history[emp_id].pop(0)

            if len(track_history[emp_id]) >= 2:
                prev_x = track_history[emp_id][-2][0]
                curr_x = track_history[emp_id][-1][0]

                # ENTRY LOGIC
                if prev_x >= DOOR_LINE_X and curr_x < DOOR_LINE_X:
                    if employee_state.get(emp_id) == "lobby":
                        timestamp = get_timestamp(frame)
                        date_str, time_str = parse_datetime(timestamp)
                        
                        # Update dictionary instead of appending to list
                        if emp_id not in attendance_records:
                            attendance_records[emp_id] = {
                                "Employee_ID": emp_id,
                                "Entry_Time": time_str,
                                "Entry_Date": date_str,
                                "Exit_Time": "-",
                                "Exit_Date": "-"
                            }
                        else:
                            attendance_records[emp_id]["Entry_Time"] = time_str
                            attendance_records[emp_id]["Entry_Date"] = date_str
                            attendance_records[emp_id]["Exit_Time"] = "-" # Reset exit if they re-enter
                            attendance_records[emp_id]["Exit_Date"] = "-"
                        
                        employee_state[emp_id] = "office"
                        entry_count += 1
                        print(f"ENTRY RECORDED: Employee {emp_id} at {timestamp}")

                # EXIT LOGIC
                elif prev_x <= DOOR_LINE_X and curr_x > DOOR_LINE_X:
                    if employee_state.get(emp_id) == "office":
                        timestamp = get_timestamp(frame)
                        date_str, time_str = parse_datetime(timestamp)
                        
                        # Update existing dictionary record
                        if emp_id not in attendance_records:
                            attendance_records[emp_id] = {
                                "Employee_ID": emp_id,
                                "Entry_Time": "-",
                                "Entry_Date": "-",
                                "Exit_Time": time_str,
                                "Exit_Date": date_str
                            }
                        else:
                            attendance_records[emp_id]["Exit_Time"] = time_str
                            attendance_records[emp_id]["Exit_Date"] = date_str
                        
                        employee_state[emp_id] = "lobby"
                        exit_count += 1
                        print(f"EXIT RECORDED: Employee {emp_id} at {timestamp}")

    # Draw ROI
    cv2.rectangle(frame, (ROI["x1"], ROI["y1"]), (ROI["x2"], ROI["y2"]), (255, 0, 0), 2)
    cv2.line(frame, (DOOR_LINE_X, ROI["y1"]), (DOOR_LINE_X, ROI["y2"]), (0, 0, 255), 2)

    # ---------------------------------------------------------
    # UI: TRANSPARENT BACKGROUND FOR COUNTERS
    # ---------------------------------------------------------
    # Define the coordinates for the background box
    box_x1 = FRAME_WIDTH - 270
    box_y1 = 20
    box_x2 = FRAME_WIDTH - 20
    box_y2 = 120

    # Create a copy of the frame to draw the transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

    # Blend the overlay with the original frame (alpha controls transparency: 0.5 = 50%)
    alpha = 0.5  
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw the text over the blended background
    text_x = FRAME_WIDTH - 250
    cv2.putText(frame, f"ENTRY: {entry_count}", (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"EXIT: {exit_count}", (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    # ---------------------------------------------------------

    cv2.imshow("Attendance System", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
save_attendance()

cap.release()
out.release()
cv2.destroyAllWindows()