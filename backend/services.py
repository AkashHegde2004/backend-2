import cv2
import numpy as np
import torch
from deepface import DeepFace
from ultralytics import YOLO
import pickle
import os
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from collections import deque
from datetime import datetime

# Enable OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Use 4 threads for OpenCV operations

# Config
TARGET_SIZE = (160, 160)
THRESHOLD = 0.5  # Lowered threshold for better recall
CONFIDENCE_THRESHOLD = 0.8  # Higher confidence requirement for face detection
ALLOWED_EXT = (".jpg", ".jpeg", ".png")
TEMPORAL_WINDOW = 5  # Number of frames for temporal smoothing

# Performance optimization
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if torch.cuda.is_available() else '-1'

# Load models
face_model = DeepFace.build_model("ArcFace")  # Switch to ArcFace

# Load YOLO model and move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(os.path.join(os.path.dirname(__file__), "../models/yolov8n.pt"))
if torch.cuda.is_available():
    yolo_model.to(device)

# Load eye detection model
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# In-memory embeddings
database_embeddings = []

# Temporal smoothing for face recognition
face_recognition_history: Dict[str, deque] = {}  # bbox_key -> deque of (identity, distance, timestamp)

def get_bbox_key(bbox):
    """Generate a key for bounding box to track across frames"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # Round to nearest 20 pixels for tolerance
    return f"{cx//20}_{cy//20}"



def load_embeddings():
    """Load embeddings from pickle file"""
    global database_embeddings
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings.pkl")
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            database_embeddings = pickle.load(f)
        print(f"✅ Loaded {len(database_embeddings)} student embeddings")

def train_embeddings(students_dir="uploads/students"):
    """Train face embeddings from student images with ArcFace"""
    global database_embeddings
    database_embeddings = []

    # Make sure we're using the correct path
    if not os.path.isabs(students_dir):
        students_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), students_dir)

    for idx, student_folder in enumerate(os.listdir(students_dir)):
        folder_path = os.path.join(students_dir, student_folder)
        if not os.path.isdir(folder_path):
            continue

        usn, name = student_folder.split("_", 1) if "_" in student_folder else ("UNKNOWN", student_folder)
        embeddings = []

        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith(ALLOWED_EXT):
                continue
            img_path = os.path.join(folder_path, img_name)

            try:
                # Use extract_faces with better preprocessing
                faces = DeepFace.extract_faces(
                    img_path, 
                    detector_backend='retinaface',  # Use RetinaFace for better accuracy
                    enforce_detection=False,
                    align=True
                )
                if not faces:
                    continue

                # Check face quality - reject low confidence
                if 'confidence' in faces[0] and faces[0]['confidence'] < CONFIDENCE_THRESHOLD:
                    print(f"Rejected low confidence face in {img_path}: {faces[0]['confidence']}")
                    continue

                face_img = faces[0]["face"]

                # Use DeepFace.represent with ArcFace for better accuracy
                embedding_obj = DeepFace.represent(
                    face_img, 
                    model_name="ArcFace",  # Switch to ArcFace
                    enforce_detection=False,
                    align=True
                )
                embedding = embedding_obj[0]["embedding"] if isinstance(embedding_obj, list) else embedding_obj["embedding"]
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        if embeddings:
            # Use median instead of mean for more robust embedding
            median_emb = np.median(embeddings, axis=0)
            database_embeddings.append((usn, name, median_emb))

        # Show progress
        print(f"Processed {idx + 1}/{len(os.listdir(students_dir))} folders: {student_folder}")

    # Save embeddings
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump(database_embeddings, f)

    print(f"✅ Trained {len(database_embeddings)} students with ArcFace")
    return len(database_embeddings)



def recognize_face(face_img, bbox=None):
    """Recognize a single face with temporal smoothing"""
    try:
        # Use extract_faces with better face detection
        faces = DeepFace.extract_faces(
            face_img, 
            detector_backend='retinaface',  # Use RetinaFace for better accuracy
            enforce_detection=False,
            align=True
        )
        if not faces:
            return None, None
            
        face_img_processed = faces[0]["face"]
        
        # Check face quality - reject low confidence detections
        if 'confidence' in faces[0] and faces[0]['confidence'] < CONFIDENCE_THRESHOLD:
            print(f"Rejected low confidence face: {faces[0]['confidence']}")
            return None, None
        
        # Get embedding using DeepFace with ArcFace
        embedding_obj = DeepFace.represent(
            face_img_processed, 
            model_name="ArcFace",  # Switch to ArcFace
            enforce_detection=False,
            align=True
        )
        
        if isinstance(embedding_obj, list):
            frame_emb = embedding_obj[0]["embedding"]
        else:
            frame_emb = embedding_obj["embedding"]
        
        min_dist = float("inf")
        identity = None
        second_min_dist = float("inf")
        
        print(f"Checking {len(database_embeddings)} students in database")
        for usn, name, db_emb in database_embeddings:
            dist = np.linalg.norm(frame_emb - db_emb)
            print(f"  {usn} - {name}: distance = {dist}")
            if dist < min_dist:
                second_min_dist = min_dist
                min_dist = dist
                if min_dist < THRESHOLD:  # Use the lowered threshold (0.5)
                    identity = (usn, name)
            elif dist < second_min_dist:
                second_min_dist = dist
        
        print(f"Best match: {identity}, min_dist: {min_dist}, second_min_dist: {second_min_dist}")
        
        # Apply temporal smoothing if bbox is provided
        if bbox is not None and identity is not None:
            bbox_key = get_bbox_key(bbox)
            current_time = datetime.utcnow()
            
            # Initialize history for this bbox if not exists
            if bbox_key not in face_recognition_history:
                face_recognition_history[bbox_key] = deque(maxlen=TEMPORAL_WINDOW)
            
            # Add current recognition to history
            face_recognition_history[bbox_key].append({
                'identity': identity,
                'distance': min_dist,
                'timestamp': current_time
            })
            
            # Clean up old bbox keys (older than 2 seconds)
            keys_to_remove = []
            for key, history in face_recognition_history.items():
                if len(history) > 0:
                    latest_time = history[-1]['timestamp']
                    if (current_time - latest_time).total_seconds() > 2.0:
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del face_recognition_history[key]
            
            # Perform temporal smoothing: take majority vote
            if len(face_recognition_history[bbox_key]) >= 3:  # Need at least 3 frames
                identity_counts = {}
                for record in face_recognition_history[bbox_key]:
                    rec_identity = record['identity']
                    if rec_identity not in identity_counts:
                        identity_counts[rec_identity] = {'count': 0, 'distances': []}
                    identity_counts[rec_identity]['count'] += 1
                    identity_counts[rec_identity]['distances'].append(record['distance'])
                
                # Get the most common identity
                most_common_identity = max(identity_counts.items(), key=lambda x: x[1]['count'])
                
                # Use this identity if it appears in at least 60% of recent frames
                if most_common_identity[1]['count'] >= len(face_recognition_history[bbox_key]) * 0.6:
                    identity = most_common_identity[0]
                    min_dist = np.mean(most_common_identity[1]['distances'])
                    print(f"Temporal smoothing: Using {identity} based on {most_common_identity[1]['count']}/{len(face_recognition_history[bbox_key])} frames")
        
        # Improved confidence checking
        if identity and second_min_dist != float("inf"):
            confidence_gap = second_min_dist - min_dist
            print(f"Confidence gap: {confidence_gap}")
            
            # More lenient confidence gap requirement
            if confidence_gap < 0.05:  # Reduced from 0.02
                print(f"Low confidence match - gap: {confidence_gap}, min_dist: {min_dist}")
                # Accept if the distance is very low
                if min_dist > 0.15:  # Reduced threshold
                    return None, min_dist
                print(f"Accepting match due to very low distance despite low confidence gap")
        
        if identity:
            print(f"Confident match found: {identity[0]} - {identity[1]}")
        else:
            print("No confident match found")
        
        return identity, min_dist
    except Exception as e:
        print(f"Recognition error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def classify_attentiveness(face_img):
    """Classify attentiveness using emotion detection and eye tracking"""
    try:
        # Convert to grayscale for eye detection
        if len(face_img.shape) == 3:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_img
        
        # --- Eye Tracking Component ---
        eyes = eye_cascade.detectMultiScale(gray_face, 1.3, 5)
        
        eye_score = 0
        eye_status = "normal"
        
        # Analyze eye detection results
        if len(eyes) == 0:
            eye_status = "closed"
            eye_score = -2  # Strong negative indicator
        elif len(eyes) == 1:
            eye_status = "partially_visible"
            eye_score = -1  # Moderate negative indicator
        elif len(eyes) >= 2:
            # Calculate eye openness based on area
            eye_areas = [w * h for (x, y, w, h) in eyes]
            avg_eye_area = sum(eye_areas) / len(eye_areas)
            face_area = gray_face.shape[0] * gray_face.shape[1]
            eye_ratio = avg_eye_area / face_area
            
            if eye_ratio < 0.005:
                eye_status = "small"
                eye_score = -1
            elif eye_ratio < 0.01:
                eye_status = "normal"
                eye_score = 0
            else:
                eye_status = "wide_open"
                eye_score = 1
        
        print(f"Eye detection: {len(eyes)} eyes, status: {eye_status}, score: {eye_score}")
        
        # --- Emotion Detection Component ---
        emotion_score = 0
        dominant_emotion = "neutral"
        
        try:
            # Use DeepFace to analyze emotions
            emotion_analysis = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(emotion_analysis, list):
                emotion_data = emotion_analysis[0]
            else:
                emotion_data = emotion_analysis
            
            dominant_emotion = emotion_data['dominant_emotion']
            emotion_scores = emotion_data['emotion']
            
            print(f"Emotion analysis: {dominant_emotion}")
            print(f"Emotion scores: {emotion_scores}")
            
            # Map emotions to attentiveness scores
            # Attentive emotions: happy, neutral, surprise (focused)
            # Inattentive emotions: angry, disgust, fear (distracted)
            # Drowsy emotions: sad (tired/bored)
            
            emotion_mapping = {
                'happy': 1,
                'neutral': 0.5,
                'surprise': 0.5,
                'angry': -0.5,
                'disgust': -0.5,
                'fear': -0.5,
                'sad': -1.5
            }
            
            emotion_score = emotion_mapping.get(dominant_emotion.lower(), 0)
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            # If emotion detection fails, rely more on eye tracking
            emotion_score = 0
        
        # --- Combined Scoring ---
        # Weight: Eye tracking (60%), Emotion (40%)
        combined_score = (eye_score * 0.6) + (emotion_score * 0.4)
        
        print(f"Combined score: {combined_score} (eye: {eye_score}, emotion: {emotion_score})")
        
        # Determine final attentiveness label
        if combined_score <= -1.0:
            return "Drowsy"
        elif combined_score <= -0.3:
            return "Inattentive"
        else:
            return "Attentive"
            
    except Exception as e:
        print(f"Attentiveness classification error: {e}")
        import traceback
        traceback.print_exc()
        return "Unknown"

def process_frame(frame):
    """Process frame for face detection, recognition, and attentiveness"""
    # Use GPU if available for YOLO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = yolo_model(frame, device=device, verbose=False)
    detections = []
    
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            continue
            
        # Check if the face is large enough for reliable recognition
        face_width = x2 - x1
        face_height = y2 - y1
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        
        # Face should be at least 10% of the frame in both dimensions
        if face_width < frame_width * 0.1 or face_height < frame_height * 0.1:
            print(f"Face too small for recognition: {face_width}x{face_height}")
            continue
            
        # Face should be reasonably proportioned (not too wide or tall)
        aspect_ratio = face_width / face_height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            print(f"Face aspect ratio unusual: {aspect_ratio}")
            continue
        
        # Pass bbox for temporal smoothing
        bbox = (x1, y1, x2, y2)
        identity, dist = recognize_face(face_img, bbox=bbox)
        
        if identity:
            usn, name = identity
            att_label = classify_attentiveness(face_img)
            
            detections.append({
                "usn": usn,
                "name": name,
                "bbox": bbox,
                "attentiveness": att_label
            })
            
            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{usn} - {name} | {att_label}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Detected: {usn} - {name} | {att_label} | Distance: {dist}")
        else:
            # Draw unknown face on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print("Detected unknown face")
    
    if detections:
        print(f"Frame processed with {len(detections)} detections")
    
    return frame, detections

# Initialize
load_embeddings()