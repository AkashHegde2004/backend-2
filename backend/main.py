from fastapi import FastAPI, Depends, HTTPException, File, UploadFile,Form,Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import cv2
import shutil
import os
from datetime import datetime, timedelta
from bson import ObjectId
import subprocess
import sys
from collections import deque
import threading
import time
import asyncio

from database import connect_db, close_db, teachers_collection, students_collection, sessions_collection, attendance_collection
from models import TeacherRegister, TeacherLogin, StudentCreate, StudentResponse, SessionCreate, SessionResponse
from auth import hash_password, verify_password, create_access_token, get_current_teacher
import services

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session state
active_session = None
camera = None
session_attendance = {}
attentiveness_history = {}
detection_stability: Dict[str, deque] = {}

STABILITY_WINDOW = 3
STABILITY_FRAMES = 5

# Threading - CRITICAL: Separate frame buffers for video and detection
video_frame = None  # For streaming only
detection_frame = None  # For processing only
annotated_overlay = {}  # Store detection results: {bbox: (usn, name, label)}
video_lock = threading.Lock()
detection_lock = threading.Lock()
processing_active = False
detection_thread = None
camera_thread = None

@app.on_event("startup")
async def startup():
    await connect_db()
    print("🚀 Server started!")

@app.on_event("shutdown")
async def shutdown():
    await close_db()

# ===== AUTH ROUTES =====
@app.post("/api/register")
async def register(teacher: TeacherRegister):
    existing = await teachers_collection.find_one({"email": teacher.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    teacher_doc = {
        "email": teacher.email,
        "name": teacher.name,
        "password": teacher.password,
        "created_at": datetime.utcnow().isoformat()
    }
    
    result = await teachers_collection.insert_one(teacher_doc)
    token = create_access_token({"sub": str(result.inserted_id)})
    
    return {"token": token, "teacher_id": str(result.inserted_id), "name": teacher.name}

@app.post("/api/login")
async def login(teacher: TeacherLogin):
    db_teacher = await teachers_collection.find_one({"email": teacher.email})
    if not db_teacher or teacher.password != db_teacher["password"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": str(db_teacher["_id"])})
    return {"token": token, "teacher_id": str(db_teacher["_id"]), "name": db_teacher["name"]}

# ===== STUDENT ROUTES =====
@app.get("/api/students", response_model=List[StudentResponse])
async def get_students(teacher_id: str = Depends(get_current_teacher)):
    students = await students_collection.find({"teacher_id": teacher_id}).to_list(100)
    return [
        StudentResponse(
            id=str(s["_id"]),
            name=s["name"],
            usn=s["usn"],
            semester=s["semester"],
            subject=s["subject"],
            department=s["department"],
            created_at=s["created_at"]
        ) for s in students
    ]

@app.post("/api/students/import")
async def import_students(
    student_ids: List[str] = Body(...),
    teacher_id: str = Depends(get_current_teacher)
):
    """Assign imported students to the current teacher."""
    try:
        # Update the teacher_id for the selected students
        result = await students_collection.update_many(
            {"_id": {"$in": [ObjectId(student_id) for student_id in student_ids]}},
            {"$set": {"teacher_id": teacher_id}}
        )
        return {"message": f"Successfully imported {result.modified_count} students."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import students: {str(e)}")

@app.get("/api/students/search")
async def search_students(
    query: str = "",
    teacher_id: str = Depends(get_current_teacher)
):
    """Search all students globally by name or USN (no teacher filtering)."""
    try:
        if query.strip():
            # Search by name or USN (case-insensitive)
            search_filter = {
                "$or": [
                    {"name": {"$regex": query, "$options": "i"}},
                    {"usn": {"$regex": query, "$options": "i"}}
                ]
            }
        else:
            # No query: return all students
            search_filter = {}

        students_list = await students_collection.find(search_filter).to_list(100)

        return [
            StudentResponse(
                id=str(s["_id"]),
                name=s["name"],
                usn=s["usn"],
                semester=s["semester"],
                subject=s["subject"],
                department=s["department"],
                created_at=s["created_at"]
            )
            for s in students_list
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/students")
async def create_student(
    name: str = Form(...),
    usn: str = Form(...),
    semester: str = Form(...),
    subject: str = Form(...),
    department: str = Form(...),
    images: List[UploadFile] = File(...),
    teacher_id: str = Depends(get_current_teacher)
):
    existing = await students_collection.find_one({"usn": usn})
    if existing:
        raise HTTPException(status_code=400, detail="USN already exists")
    
    student_folder = f"uploads/students/{usn}_{name}"
    os.makedirs(student_folder, exist_ok=True)
    
    for idx, image in enumerate(images):
        file_path = f"{student_folder}/{idx}_{image.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
    
    student_doc = {
        "teacher_id": teacher_id,
        "name": name,
        "usn": usn,
        "semester": semester,
        "subject": subject,
        "department": department,
        "image_path": student_folder,
        "created_at": datetime.utcnow().isoformat()
    }
    
    result = await students_collection.insert_one(student_doc)
    
    return {"id": str(result.inserted_id), "message": "Student created successfully"}

@app.post("/api/train-model")
async def train_model(teacher_id: str = Depends(get_current_teacher)):
    try:
        count = services.train_embeddings()
        return {"message": f"Model trained with {count} students using ArcFace", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train-improved-model")
async def train_improved_model(teacher_id: str = Depends(get_current_teacher)):
    try:
        import improved_training
        count = improved_training.train_embeddings_improved()
        services.load_embeddings()
        return {"message": f"Improved model trained with {count} students", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train-cnn-model")
async def train_cnn_model(teacher_id: str = Depends(get_current_teacher)):
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "cnn_training.py")
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            services.load_cnn_model()
            return {"message": "CNN model trained successfully", "output": result.stdout}
        else:
            raise HTTPException(status_code=500, detail=f"CNN training failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="CNN training timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def camera_capture_thread():
    """Dedicated thread for camera capture - runs at full speed"""
    global video_frame, detection_frame, processing_active
    
    print("📹 Camera thread started")
    frame_count = 0
    
    while processing_active and camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            time.sleep(0.001)
            continue
        
        frame_count += 1
        
        # Update video frame (for streaming) - ALWAYS
        with video_lock:
            video_frame = frame.copy()
        
        # Update detection frame every 10th frame (3fps detection)
        if frame_count % 10 == 0:
            with detection_lock:
                detection_frame = frame.copy()
        
        # No sleep - capture as fast as possible
    
    print("📹 Camera thread stopped")

def process_detections_thread():
    """Background thread for processing detection - independent of video"""
    global detection_frame, annotated_overlay, session_attendance, attentiveness_history, detection_stability, processing_active
    
    print("🔍 Detection thread started")
    detection_count = 0
    last_detection_time = time.time()
    
    while processing_active:
        try:
            # Get frame to process
            with detection_lock:
                if detection_frame is None:
                    time.sleep(0.05)
                    continue
                frame_to_process = detection_frame.copy()
                detection_frame = None  # Clear so we don't process same frame twice
            
            detection_count += 1
            
            # Show FPS every 10 detections
            if detection_count % 10 == 0:
                elapsed = time.time() - last_detection_time
                fps = 10 / elapsed if elapsed > 0 else 0
                print(f"🔍 DETECTION FPS: {fps:.2f}")
                last_detection_time = time.time()
            
            # Process frame (this is slow - but doesn't block video!)
            _, detections = services.process_frame(frame_to_process)
            
            # Update overlay with detection results
            new_overlay = {}
            current_time = datetime.utcnow()
            detected_usns = set()
            
            for detection in detections:
                usn = detection["usn"]
                detected_usns.add(usn)
                bbox = detection["bbox"]
                
                # Store overlay info
                new_overlay[bbox] = (usn, detection["name"], detection["attentiveness"])
                
                # Update stability tracking
                if usn not in detection_stability:
                    detection_stability[usn] = deque(maxlen=STABILITY_FRAMES)
                detection_stability[usn].append(current_time)
            
            # Update shared overlay
            with detection_lock:
                annotated_overlay.clear()
                annotated_overlay.update(new_overlay)
            
            # Clean up old stability records
            usns_to_remove = []
            for usn in detection_stability:
                if usn not in detected_usns:
                    detection_stability[usn] = deque(
                        [ts for ts in detection_stability[usn] if (current_time - ts).total_seconds() <= 2.0],
                        maxlen=STABILITY_FRAMES
                    )
                    if len(detection_stability[usn]) == 0:
                        usns_to_remove.append(usn)
            
            for usn in usns_to_remove:
                del detection_stability[usn]
            
            # Track attendance for stable detections
            for detection in detections:
                usn = detection["usn"]
                
                if usn in detection_stability and len(detection_stability[usn]) >= STABILITY_WINDOW:
                    if usn not in session_attendance:
                        session_attendance[usn] = {
                            "name": detection["name"],
                            "first_seen": current_time.isoformat(),
                            "scores": []
                        }
                        attentiveness_history[usn] = {
                            "history": [],
                            "drowsy_count": 0,
                            "inattentive_count": 0,
                            "attentive_count": 0
                        }
                        print(f"✅ New student: {usn} - {detection['name']}")
                    
                    attentiveness_label = detection["attentiveness"]
                    attentiveness_history[usn]["history"].append({
                        "time": current_time,
                        "label": attentiveness_label
                    })
                    
                    cutoff_time = current_time - timedelta(seconds=30)
                    attentiveness_history[usn]["history"] = [
                        record for record in attentiveness_history[usn]["history"]
                        if record["time"] > cutoff_time
                    ]
                    
                    adjusted_label = attentiveness_label
                    
                    if attentiveness_label == "Drowsy":
                        attentiveness_history[usn]["drowsy_count"] += 1
                        attentiveness_history[usn]["inattentive_count"] = max(0, attentiveness_history[usn]["inattentive_count"] - 0.3)
                        attentiveness_history[usn]["attentive_count"] = max(0, attentiveness_history[usn]["attentive_count"] - 0.3)
                    elif attentiveness_label == "Inattentive":
                        attentiveness_history[usn]["inattentive_count"] += 1
                        attentiveness_history[usn]["drowsy_count"] = max(0, attentiveness_history[usn]["drowsy_count"] - 0.3)
                        attentiveness_history[usn]["attentive_count"] = max(0, attentiveness_history[usn]["attentive_count"] - 0.3)
                    else:
                        attentiveness_history[usn]["attentive_count"] += 1
                        attentiveness_history[usn]["drowsy_count"] = max(0, attentiveness_history[usn]["drowsy_count"] - 0.5)
                        attentiveness_history[usn]["inattentive_count"] = max(0, attentiveness_history[usn]["inattentive_count"] - 0.5)
                    
                    if attentiveness_history[usn]["drowsy_count"] > 5:
                        adjusted_label = "Drowsy"
                    elif attentiveness_history[usn]["inattentive_count"] > 8:
                        adjusted_label = "Inattentive"
                    elif attentiveness_history[usn]["attentive_count"] > 5:
                        adjusted_label = "Attentive"
                    
                    if len(attentiveness_history[usn]["history"]) >= 5:
                        recent_labels = [rec["label"] for rec in attentiveness_history[usn]["history"][-5:]]
                        label_counts = {}
                        for label in recent_labels:
                            label_counts[label] = label_counts.get(label, 0) + 1
                        
                        max_label = max(label_counts.items(), key=lambda x: x[1])
                        if max_label[1] >= 3:
                            adjusted_label = max_label[0]
                    
                    session_attendance[usn]["scores"].append({
                        "time": current_time.isoformat(),
                        "label": adjusted_label
                    })
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            time.sleep(0.1)
    
    print("🔍 Detection thread stopped")

# ===== SESSION ROUTES =====
@app.post("/api/session/start")
async def start_session(
    session_data: SessionCreate,
    teacher_id: str = Depends(get_current_teacher)
):
    global active_session, camera, session_attendance, attentiveness_history, detection_stability
    global video_frame, detection_frame, annotated_overlay, processing_active, detection_thread, camera_thread
    
    if active_session:
        raise HTTPException(status_code=400, detail="Session already active")
    
    print("\n" + "="*50)
    print("📹 Starting session...")
    print("="*50)
    
    session_doc = {
        "teacher_id": teacher_id,
        "subject": session_data.subject,
        "semester": session_data.semester,
        "department": session_data.department,
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None,
        "status": "active"
    }
    
    result = await sessions_collection.insert_one(session_doc)
    active_session = str(result.inserted_id)
    session_attendance = {}
    attentiveness_history = {}
    detection_stability = {}
    video_frame = None
    detection_frame = None
    annotated_overlay = {}
    
    # Start camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Start threads
    processing_active = True
    
    camera_thread = threading.Thread(target=camera_capture_thread, daemon=True, name="CameraThread")
    camera_thread.start()
    
    detection_thread = threading.Thread(target=process_detections_thread, daemon=True, name="DetectionThread")
    detection_thread.start()
    
    time.sleep(0.2)
    print(f"✅ Threads started: Camera={camera_thread.is_alive()}, Detection={detection_thread.is_alive()}")
    print("="*50 + "\n")
    
    return {"session_id": active_session, "message": "Session started"}

@app.post("/api/session/stop")
async def stop_session(teacher_id: str = Depends(get_current_teacher)):
    global active_session, camera, session_attendance, attentiveness_history, detection_stability
    global processing_active, detection_thread, camera_thread, video_frame, detection_frame, annotated_overlay
    
    if not active_session:
        raise HTTPException(status_code=400, detail="No active session")
    
    print("\n🛑 Stopping session...")
    
    # Stop threads
    processing_active = False
    if camera_thread:
        camera_thread.join(timeout=1.0)
    if detection_thread:
        detection_thread.join(timeout=2.0)
    
    # Stop camera
    if camera:
        camera.release()
        camera = None
    
    await sessions_collection.update_one(
        {"_id": ObjectId(active_session)},
        {"$set": {"end_time": datetime.utcnow().isoformat(), "status": "completed"}}
    )
    
    attendance_records = []
    for usn, data in session_attendance.items():
        attendance_records.append({
            "session_id": active_session,
            "student_usn": usn,
            "student_name": data["name"],
            "first_seen": data["first_seen"],
            "attentiveness_scores": data["scores"]
        })
    
    if attendance_records:
        await attendance_collection.insert_many(attendance_records)
    
    print(f"✅ Session stopped. Saved {len(attendance_records)} students\n")
    
    session_id = active_session
    active_session = None
    session_attendance = {}
    attentiveness_history = {}
    detection_stability = {}
    video_frame = None
    detection_frame = None
    annotated_overlay = {}
    
    return {"session_id": session_id, "message": "Session stopped", "attendance_count": len(attendance_records)}

@app.get("/api/session/video-feed")
async def video_feed():
    def generate():
        global video_frame, annotated_overlay
        
        print("📺 Video feed connected")
        
        # JPEG encoding
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        
        frame_count = 0
        last_fps_time = time.time()
        
        while processing_active:
            # Get current video frame
            with video_lock:
                if video_frame is None:
                    time.sleep(0.01)
                    continue
                current_frame = video_frame.copy()
            
            # Draw annotations from detection thread
            with detection_lock:
                for bbox, (usn, name, label) in annotated_overlay.items():
                    x1, y1, x2, y2 = bbox
                    color = (0, 255, 0) if label == "Attentive" else (0, 165, 255) if label == "Inattentive" else (0, 0, 255)
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(current_frame, f"{usn} - {name} | {label}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # FPS monitoring
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - last_fps_time
                fps = 30 / elapsed if elapsed > 0 else 0
                print(f"📹 VIDEO FPS: {fps:.2f}")
                last_fps_time = time.time()
            
            # Encode and stream
            ret, buffer = cv2.imencode('.jpg', current_frame, encode_param)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Target 30fps
            time.sleep(0.033)
        
        print("📺 Video feed disconnected")
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ===== ANALYTICS ROUTES =====
@app.get("/api/sessions")
async def get_sessions(teacher_id: str = Depends(get_current_teacher)):
    sessions = await sessions_collection.find({"teacher_id": teacher_id}).sort("start_time", -1).to_list(100)
    return [
        {
            "id": str(s["_id"]),
            "subject": s["subject"],
            "semester": s["semester"],
            "department": s["department"],
            "start_time": s["start_time"],
            "end_time": s.get("end_time"),
            "status": s["status"]
        } for s in sessions
    ]

@app.get("/api/session/current-attendance")
async def get_current_attendance(teacher_id: str = Depends(get_current_teacher)):
    global session_attendance
    
    if not active_session:
        raise HTTPException(status_code=400, detail="No active session")
    
    attendance_records = []
    for usn, data in session_attendance.items():
        attendance_records.append({
            "student_usn": usn,
            "student_name": data["name"],
            "first_seen": data["first_seen"],
            "attentiveness_scores": data["scores"]
        })
    
    return attendance_records

@app.get("/api/attendance/{session_id}")
async def get_attendance(session_id: str, teacher_id: str = Depends(get_current_teacher)):
    records = await attendance_collection.find({"session_id": session_id}).to_list(100)
    for record in records:
        record["_id"] = str(record["_id"])
    return records

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)