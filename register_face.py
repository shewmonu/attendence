import cv2
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
import os
import time

# =========================
# CONFIG
# =========================
SAMPLES_NEEDED = 10       
DET_SCORE_MIN  = 0.75     # Minimum acceptable for CCTV
DISPLAY_WIDTH  = 900
DISPLAY_HEIGHT = 600
MIN_SAMPLE_INTERVAL = 0.3 # 10 samples over ~3s of walking

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
# INSIGHTFACE
# =========================
print("[Setup] Loading InsightFace buffalo_l model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[Setup] Model ready.\n")

# =========================
# HELPERS
# =========================
def employee_exists(eid: int) -> bool:
    cursor.execute("SELECT eid FROM employees WHERE eid = %s", (eid,))
    return cursor.fetchone() is not None

def already_has_embedding(eid: int) -> bool:
    cursor.execute("SELECT face_embedding FROM employees WHERE eid = %s", (eid,))
    row = cursor.fetchone()
    return row is not None and row[0] is not None

def save_embedding(eid: int, name: str, avg_emb: np.ndarray) -> None:
    emb_bytes = avg_emb.astype(np.float32).tobytes()
    cursor.execute("""
        INSERT INTO employees (eid, name, face_embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (eid)
        DO UPDATE SET
            name           = EXCLUDED.name,
            face_embedding = EXCLUDED.face_embedding
    """, (eid, name, psycopg2.Binary(emb_bytes)))
    conn.commit()

def draw_ui(frame, collected, needed, name, eid, best_score, status, status_color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, f"Registering: {name} (EID {eid})",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Samples: {collected}/{needed}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 18
    pct = min(collected / needed, 1.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + bar_h), (0, 200, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

    score_color = (0, 220, 80) if best_score >= DET_SCORE_MIN else (0, 140, 255)
    cv2.putText(frame, f"Det score: {best_score:.2f}",
                (bar_x, bar_y + bar_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 1)

    cv2.putText(frame, status, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    return frame

# =========================
# ENROLLMENT FLOW
# =========================
def get_employee_info():
    print("=" * 50)
    print("   VIDEO FILE FACE REGISTRATION")
    print("=" * 50)

    while True:
        try:
            eid = int(input("\nEnter Employee EID: ").strip())
        except ValueError:
            print("  [!] EID must be a number. Try again.")
            continue
            
        if not employee_exists(eid):
            print(f"  [!] EID {eid} does not exist in the DB. Will create a new record.")

        name = input("Enter Employee Name: ").strip()
        if not name:
            print("  [!] Name cannot be empty. Try again.")
            continue

        if already_has_embedding(eid):
            print(f"\n  [!] EID {eid} already has a face embedding.")
            choice = input("  Overwrite? (y/n): ").strip().lower()
            if choice != "y":
                print("  Enter a different EID.\n")
                continue   
                
        video_path = input("Enter Video File Path (Press Enter for 'CENTURY_DATA.mp4'): ").strip()
        if not video_path:
            video_path = "CENTURY_DATA.mp4"

        print(f"\n  -> Will register: {name} (EID {eid}) from {video_path}")
        return eid, name, video_path

def collect_embeddings(eid: int, name: str, video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return None

    samples = []
    last_sample_time = 0.0
    paused = False
    capturing = False
    
    status = "WAITING: Press 'Space' to pause/resume, 'R' to capture!"
    status_color = (0, 255, 255)

    print("\n[Video] Playing video...")
    print("        Press Spacebar to pause/resume.")
    print("        Press 'R' to start grabbing face frames.")
    print("        Press ESC to cancel.\n")

    # Pre-read the first frame so we have a clean copy to pause on
    ret, raw_frame = cap.read()
    if not ret:
        print("[Error] Video is empty or unreadable.")
        return None
    raw_frame = cv2.resize(raw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    while True:
        # Only advance the video if we are NOT paused
        if not paused:
            ret, next_frame = cap.read()
            if not ret:
                print("\n[Warning] Reached end of video before finishing capture.")
                break
            raw_frame = cv2.resize(next_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Always work on a fresh copy so drawings don't smear while paused
        frame = raw_frame.copy()
        
        faces = face_app.get(frame)
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)

        best_score = 0.0
        accepted_this_frame = False

        # Key listener
        key = cv2.waitKey(30) & 0xFF
        if key == 27:   # ESC
            print("\n[Cancelled] Aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == ord(' '):
            paused = not paused
            if not paused:
                status = "Resumed. Press 'R' to capture." if not capturing else status
                status_color = (0, 255, 255)
            else:
                status = "PAUSED: Press Space to resume, 'R' to capture!"
                status_color = (0, 255, 255)
        elif key == ord('r') or key == ord('R'):
            capturing = True
            paused = False

        for idx, face in enumerate(faces):
            score = float(face.det_score)
            if score > best_score:
                best_score = score

            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            box_color = (0, 220, 80) if score >= DET_SCORE_MIN else (0, 140, 255)
            
            if idx == 0 and capturing:
                box_color = (255, 0, 255) # Pink for targeted face
                
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), box_color, 2)
            cv2.putText(frame, f"{score:.2f}", (fx1, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1)

            if capturing and idx == 0 and score >= DET_SCORE_MIN and not accepted_this_frame:
                now = time.time()
                if now - last_sample_time >= MIN_SAMPLE_INTERVAL:
                    emb = face.embedding / np.linalg.norm(face.embedding)
                    samples.append(emb)
                    last_sample_time = now
                    accepted_this_frame = True
                    status = f"CAPTURING! Keep target on screen... ({len(samples)}/{SAMPLES_NEEDED})"
                    status_color = (0, 220, 80)

        if capturing and not faces:
            status = "Target lost! Waiting for face..."
            status_color = (0, 140, 255)

        frame = draw_ui(frame, len(samples), SAMPLES_NEEDED, name, eid, best_score, status, status_color)
        cv2.imshow("Face Registration", frame)

        if len(samples) >= SAMPLES_NEEDED:
            status = "Collection complete! Saving..."
            frame = draw_ui(frame, len(samples), SAMPLES_NEEDED, name, eid, best_score, status, (0, 220, 80))
            cv2.imshow("Face Registration", frame)
            cv2.waitKey(1200)
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < SAMPLES_NEEDED:
        print(f"[Error] Only collected {len(samples)} samples. Need {SAMPLES_NEEDED}.")
        return None

    avg_emb = np.mean(np.stack(samples), axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    return avg_emb

# =========================
# MAIN
# =========================
def main():
    try:
        eid, name, video_path = get_employee_info()
    except (KeyboardInterrupt, EOFError):
        print("\n[Cancelled]")
        return

    avg_emb = collect_embeddings(eid, name, video_path)

    if avg_emb is None:
        print("[Failed] No embedding saved.")
        return

    print(f"\n[DB] Saving averaged embedding for {name} (EID {eid})...")
    save_embedding(eid, name, avg_emb)
    print(f"[Done] ✓ {name} (EID {eid}) registered successfully.\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        cursor.close()
        conn.close()