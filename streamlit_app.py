import streamlit as st
import face_recognition # המנוע הקל למציאת פנים
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import os
import zipfile
import random
import cv2
from rembg import remove
import json
from datetime import datetime
import pandas as pd
from io import BytesIO
import tensorflow as tf

# הגדרות עמוד
st.set_page_config(
    page_title="Smart Attendance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# אתחול Session State
if "mode" not in st.session_state:
    st.session_state.mode = "upload"
if "collected_photos" not in st.session_state:
    st.session_state.collected_photos = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "absence_counter" not in st.session_state:
    st.session_state.absence_counter = {}
if "student_roster" not in st.session_state:
    st.session_state.student_roster = []

ABSENCE_THRESHOLD = 3

# --- CSS מעוצב (העיצוב המקורי שלך) ---
css = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap"/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0"/>
<style>
* { font-family: 'Space Grotesk', sans-serif !important; }
.material-symbols-outlined { font-family: 'Material Symbols Outlined' !important; font-size: 22px; display: inline-block; vertical-align: middle; }
@keyframes scanLine { 0% { top: 0%; opacity: 1; } 100% { top: 100%; opacity: 0.3; } }
.stApp { background: #f0eef4 !important; }
.main-header { display: flex; align-items: center; gap: 14px; padding: 1.5rem 0 1rem; border-bottom: 1px solid #e4dff0; margin-bottom: 1.5rem; }
.header-icon { width: 52px; height: 52px; background: linear-gradient(135deg, #b8a9c9, #9585b0); border-radius: 14px; display: flex; align-items: center; justify-content: center; }
.header-title { font-size: 28px; font-weight: 700; background: linear-gradient(90deg, #6b5a8a, #9585b0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.scan-container { position: relative; width: 100%; }
.scan-line { position: absolute; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, transparent, #b8a9c9, #c4b8d8, #b8a9c9, transparent); animation: scanLine 1.5s infinite; }
.stat-card { background: #fff; border: 1px solid #e4dff0; border-radius: 12px; padding: 16px; text-align: center; }
.stat-val { font-size: 28px; font-weight: 700; }
.stat-green { color: #68b88a; } .stat-red { color: #d4707a; } .stat-gold { color: #b8a9c9; }
.sidebar-student { display: flex; align-items: center; gap: 8px; padding: 8px; background: #f0eef4; border-radius: 8px; margin-bottom: 6px; font-size: 13px; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# --- נתיבים וקבצים ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "My_Classmates_small.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "My_Classmates")
MODEL_PATH = os.path.join(BASE_DIR, "my_siamese3_model.h5")

if not os.path.exists(EXTRACT_PATH) and os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = os.path.join(EXTRACT_PATH, "content", "My_Classmates_small")
if not os.path.exists(REFERENCE_DIR):
    REFERENCE_DIR = EXTRACT_PATH # גיבוי למקרה שהמבנה שונה

# --- פונקציות עזר ---
def load_roster():
    roster_file = os.path.join(BASE_DIR, "student_roster.json")
    if os.path.exists(roster_file):
        with open(roster_file, "r") as f:
            return json.load(f)
    return [d for d in os.listdir(REFERENCE_DIR) if os.path.isdir(os.path.join(REFERENCE_DIR, d))]

def save_roster(roster):
    with open(os.path.join(BASE_DIR, "student_roster.json"), "w") as f:
        json.dump(roster, f)

if not st.session_state.student_roster:
    st.session_state.student_roster = load_roster()

# --- טעינת המודל שלך (my_siamese3) ---
@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"Model file {MODEL_PATH} not found!")
        return None

embedding_model = load_my_model()

def get_embedding(face_img):
    """מכין תמונה למודל ומחלץ Embedding"""
    # המודל שלך מצפה ל-128x128. ודאי שזה הגודל הנכון.
    img_array = np.array(face_img.resize((128, 128))).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    embedding = embedding_model.predict(img_array, verbose=0)[0]
    return embedding

def cosine_distance(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@st.cache_resource
def load_reference_embeddings():
    embeddings = {}
    if not os.path.exists(REFERENCE_DIR): return {}
    for student in os.listdir(REFERENCE_DIR):
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            student_embs = []
            for file in os.listdir(student_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        img = Image.open(os.path.join(student_path, file)).convert("RGB")
                        # זיהוי פנים בסיסי בשביל הקרופ הראשוני
                        img_arr = np.array(img)
                        locs = face_recognition.face_locations(img_arr)
                        if locs:
                            t, r, b, l = locs[0]
                            face_crop = img.crop((l, t, r, b))
                            student_embs.append(get_embedding(face_crop))
                    except: continue
            if student_embs: embeddings[student] = student_embs
    return embeddings

reference_embeddings = load_reference_embeddings()

# --- פונקציית הזיהוי המרכזית ---
def recognize_faces(image_pil, threshold=0.5):
    scan_placeholder = st.empty()
    scan_placeholder.markdown('<div class="scan-container"><div class="scan-line"></div><div style="background:#ebe8f240;padding:2rem;text-align:center;border-radius:12px;">Scanning...</div></div>', unsafe_allow_html=True)
    
    progress = st.progress(0, text="Detecting faces...")
    img_rgb = np.array(image_pil.convert("RGB"))
    
    # שימוש ב-face_recognition למציאת מיקומים (החלק המהיר)
    face_locations = face_recognition.face_locations(img_rgb, model="hog")
    
    scan_placeholder.empty()
    present_students = {}
    recognized_faces = []

    for i, (top, right, bottom, left) in enumerate(face_locations):
        progress.progress(int(100 * (i+1)/max(len(face_locations),1)), text=f"Analyzing face {i+1}...")
        
        # חיתוך ושימוש במודל שלך
        face_crop = image_pil.crop((left, top, right, bottom))
        current_emb = get_embedding(face_crop)
        
        best_name, best_dist = "Unknown", 1.0
        for name, ref_embs in reference_embeddings.items():
            dists = [cosine_distance(current_emb, r) for r in ref_embs]
            min_dist = min(dists)
            if min_dist < best_dist:
                best_dist = min_dist
                best_name = name
        
        if best_dist > threshold: best_name = "Unknown"
        
        box = [left, top, right - left, bottom - top]
        if best_name != "Unknown" and best_name not in present_students:
            present_students[best_name] = {"img": face_crop, "dist": best_dist}
        
        recognized_faces.append({"name": best_name, "box": box, "dist": best_dist})

    progress.empty()
    
    # ציור על התמונה
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    for f in recognized_faces:
        x, y, w, h = f["box"]
        color = "#68b88a" if f["name"] != "Unknown" else "#d4707a"
        draw.rectangle([x, y, x+w, y+h], outline=color, width=4)
        draw.text((x, y-20), f["name"], fill=color)

    st.image(img_draw, use_column_width=True)
    
    # סטטיסטיקות
    known_present = list(present_students.keys())
    missing = [s for s in st.session_state.student_roster if s not in known_present]
    
    st.session_state.last_results = {"present": known_present, "missing": missing, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Present", len(known_present))
    col2.metric("Absent", len(missing))
    col3.metric("Rate", f"{int(len(known_present)/len(st.session_state.student_roster)*100)}%")
    
    return present_students

# --- UI ראשי ---
st.markdown('<div class="main-header"><div class="header-icon"><span class="material-symbols-outlined" style="color:white">face</span></div><div class="header-title">Smart Attendance System</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📋 Class Roster")
    for s in st.session_state.student_roster:
        st.markdown(f'<div class="sidebar-student"><span class="material-symbols-outlined">person</span> {s}</div>', unsafe_allow_html=True)
    
    st.divider()
    conf_thresh = st.slider("Sensitivity (Threshold)", 0.1, 0.9, 0.5)

# טאבים למצבי עבודה
tab1, tab2 = st.tabs(["📤 Upload Photo", "📸 Live Camera"])

with tab1:
    uploaded_file = st.file_uploader("Choose a classroom photo", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        results = recognize_faces(img, threshold=conf_thresh)
        if results:
            st.success("Attendance scan complete!")
            cols = st.columns(5)
            for i, (name, data) in enumerate(results.items()):
                with cols[i % 5]:
                    st.image(data["img"], caption=name)

with tab2:
    cam_img = st.camera_input("Take a photo of the class")
    if cam_img:
        img = Image.open(cam_img)
        recognize_faces(img, threshold=conf_thresh)
