import cv2
import numpy as np
import supervision as sv
import psycopg2
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
import os

# =========================
# DB CONNECTION (Using your existing .env)
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
# INSIGHTFACE SETUP
# =========================
print("[Setup] Loading InsightFace buffalo_l model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Helper to match Face Bounding Boxes with Tracker Bounding Boxes
def _iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def enroll_from_cctv(camera_source="CENTURY_DATA3.mp4", samples_to_keep=15):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Cannot open video: {camera_source}")
        return

    # Initialize ByteTrack
    enroll_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )

    temp_tracks = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[Phase 1] Auto-Scanning {total_frames} frames from video...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        if frame_idx % 30 == 0: 
            print(f"\rScanning Frame {frame_idx}/{total_frames}...", end="", flush=True)

        raw_faces = face_app.get(frame)
        if not raw_faces: continue

        xyxy = np.array([f.bbox for f in raw_faces])
        confidence = np.array([float(f.det_score) for f in raw_faces])
        
        # Filter out blurry faces to maintain database quality
        valid_idx = confidence > 0.60
        if not any(valid_idx): continue
        
        tracked = enroll_tracker.update_with_detections(
            sv.Detections(xyxy=xyxy[valid_idx], confidence=confidence[valid_idx])
        )

        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            bbox = tracked.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox)

            best_fi, best_iou = None, 0.3
            for fi, rf in enumerate(raw_faces):
                iv = _iou(bbox, rf.bbox)
                if iv > best_iou: 
                    best_iou = iv
                    best_fi = fi

            if best_fi is not None:
                if tid not in temp_tracks: 
                    temp_tracks[tid] = {'embeddings': [], 'face_crops': []}
                
                tt = temp_tracks[tid]
                tt['embeddings'].append(raw_faces[best_fi].embedding / np.linalg.norm(raw_faces[best_fi].embedding))

                # Save face image crop for manual identification later
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    tt['face_crops'].append(frame[cy1:cy2, cx1:cx2].copy())

    cap.release()
    print(f"\n[Phase 1 Complete] Found {len(temp_tracks)} distinct tracked people.")
    if not temp_tracks: return

    # Limit samples evenly
    for tid in temp_tracks:
        tt = temp_tracks[tid]
        n = len(tt['embeddings'])
        if n > samples_to_keep:
            idx = [int(i * n / samples_to_keep) for i in range(samples_to_keep)]
            tt['embeddings'] = [tt['embeddings'][j] for j in idx]
            tt['face_crops'] = [tt['face_crops'][j] for j in idx if j < len(tt['face_crops'])]

    # =======================================================
    # PHASE 2: MANUAL IDENTIFICATION + DATABASE INSERTION
    # =======================================================
    print("\n" + "="*50)
    print("   PHASE 2: IDENTIFY FACES")
    print("="*50)
    print("Look at the image popup, then type their info in the console.")
    
    saved_count = 0

    for tid in sorted(temp_tracks.keys()):
        data = temp_tracks[tid]
        
        if len(data['embeddings']) < 5: 
            continue # Skip garbage tracks with barely any face data

        print(f"\n--- Track #{tid}: {len(data['embeddings'])} matched face frames ---")
        
        if data['face_crops']:
            mid_crop = data['face_crops'][len(data['face_crops']) // 2]
            ch, cw = mid_crop.shape[:2]
            if ch < 150 or cw < 150:
                mid_crop = cv2.resize(mid_crop, (200, int(200 * ch / cw)))
                
            # 🔥 THE FIX: Force the popup window to the absolute front!
            cv2.namedWindow("Who is this?", cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty("Who is this?", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow("Who is this?", mid_crop)
            cv2.waitKey(1) 

        try:
            eid_str = input(f"   Enter EID for Track #{tid} (Press Enter to skip): ").strip()
            if not eid_str: 
                print("   Skipped.")
                continue
            eid = int(eid_str)
            name = input(f"   Enter Name: ").strip()
        except ValueError:
            print("   [!] Invalid EID. Skipping.")
            continue
            
        cv2.destroyAllWindows() 

       
        # ==========================================
        # 🔥 THE NEW "GROWING GALLERY" FIX
        # ==========================================
        # 1. Check if this person already has saved frames in the database
        cursor.execute("SELECT face_embedding FROM employees WHERE eid = %s", (eid,))
        row = cursor.fetchone()
        
        new_embs = np.stack(data['embeddings']).astype(np.float32)
        
        if row and row[0] is not None:
            # Unpack the existing frames and combine them with the new frames!
            existing_embs = np.frombuffer(bytes(row[0]), dtype=np.float32).reshape(-1, 512)
            all_embs = np.vstack((existing_embs, new_embs))
            print(f"   [+] Appending angles. Gallery grew from {len(existing_embs)} to {len(all_embs)} frames!")
        else:
            all_embs = new_embs

        emb_bytes = all_embs.tobytes()

        # Database Insertion
        cursor.execute("""
            INSERT INTO employees (eid, name, face_embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (eid)
            DO UPDATE SET
                name           = EXCLUDED.name,
                face_embedding = EXCLUDED.face_embedding
        """, (eid, name, psycopg2.Binary(emb_bytes)))
        
        conn.commit()
        saved_count += 1
        print(f"   ✅ Saved {name} (EID {eid}) to database! (Total Gallery Size: {len(all_embs)} frames)")

    cursor.close()
    conn.close()
    cv2.destroyAllWindows()
    print(f"\nDone! Successfully enrolled {saved_count} people from CCTV.")

if __name__ == "__main__":
    enroll_from_cctv(camera_source="CENTURY_DATA.mp4", samples_to_keep=15)