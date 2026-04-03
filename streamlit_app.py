import streamlit as st
from deepface import DeepFace
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

# ===== NEW: TensorFlow + MobileNetV2 =====
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Smart Attendance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STATE ----------------
if "mode" not in st.session_state:
    st.session_state.mode = "upload"
if "collected_photos" not in st.session_state:
    st.session_state.collected_photos = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "absence_counter" not in st.session_state:
    st.session_state.absence_counter = {}

ABSENCE_THRESHOLD = 3

# ---------------- STYLE (unchanged) ----------------
css = """<style>/* ... keep your original CSS here ... */</style>"""
button_css = """<style>/* ... keep your original button CSS here ... */</style>"""
st.markdown(css, unsafe_allow_html=True)
st.markdown(button_css, unsafe_allow_html=True)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "My_Classmates_small.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "My_Classmates")
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = os.path.join(EXTRACT_PATH, "content", "My_Classmates_small")
ROSTER_FILE = os.path.join(BASE_DIR, "student_roster.json")

# ---------------- ROSTER ----------------
def load_roster():
    if os.path.exists(ROSTER_FILE):
        with open(ROSTER_FILE, "r") as f:
            return json.load(f)
    return ['Maayan','Tomer','Roei','Zohar','Ilay']

def save_roster(roster):
    with open(ROSTER_FILE, "w") as f:
        json.dump(roster, f)

def update_absences(missing_students):
    for name in missing_students:
        st.session_state.absence_counter[name] = st.session_state.absence_counter.get(name, 0) + 1
    return st.session_state.absence_counter

def export_to_excel(present, missing, date_str):
    output = BytesIO()
    rows = []
    for name in present:
        rows.append({"Name": name, "Status": "Present", "Date": date_str})
    for name in missing:
        rows.append({"Name": name, "Status": "Absent", "Date": date_str})
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    return output.getvalue()

if "student_roster" not in st.session_state:
    st.session_state.student_roster = load_roster()
STUDENT_ROSTER = st.session_state.student_roster

# ---------------- MODEL (YOUR EMBEDDING) ----------------
def build_pro_embedding():
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation=None),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2_norm")
    ], name="MobileNetV2_Siamese_Pro")

    return model

@st.cache_resource
def load_embedding_model():
    model = build_pro_embedding()
    weights_path = os.path.join(BASE_DIR, "face_encoder.weights.h5")
    model.load_weights(weights_path)
    return model

embedding_model = load_embedding_model()

def get_embedding(img_pil: Image.Image):
    img = img_pil.resize((128, 128))
    img = np.array(img).astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    emb = embedding_model.predict(img, verbose=0)[0]
    return emb

# ---------------- REFERENCES ----------------
@st.cache_resource
def load_reference_embeddings():
    embeddings = {}
    for student in os.listdir(REFERENCE_DIR):
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            student_embeddings = []
            for file in os.listdir(student_path):
                if file.lower().endswith((".jpg",".jpeg",".png",".jfif")):
                    img_path = os.path.join(student_path, file)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        emb = get_embedding(img)
                        student_embeddings.append(emb)
                    except:
                        pass
            if student_embeddings:
                embeddings[student] = student_embeddings
    return embeddings

@st.cache_resource
def load_reference_photos():
    photos = {}
    for student in STUDENT_ROSTER:
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            files = [f for f in os.listdir(student_path)
                     if f.lower().endswith((".jpg",".jpeg",".png",".jfif"))]
            if files:
                img_path = os.path.join(student_path, files[0])
                photos[student] = Image.open(img_path).convert("RGB")
    return photos

reference_embeddings = load_reference_embeddings()
reference_photos = load_reference_photos()

# ---------------- UI HEADER (unchanged) ----------------
st.markdown("""
<div class="main-header">
    <div class="header-icon">
        <span class="material-symbols-outlined">face_unlock</span>
    </div>
    <div>
        <div class="header-title">Smart Attendance</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR (unchanged logic) ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Class roster</div>', unsafe_allow_html=True)
    for s in STUDENT_ROSTER:
        count = st.session_state.absence_counter.get(s, 0)
        badge = f" ({count}x)" if count > 0 else ""
        st.write(f"{s}{badge}")

    if st.session_state.last_results is not None:
        results = st.session_state.last_results
        excel_data = export_to_excel(results["present"], results["missing"], results["date"])
        st.download_button(
            label="Export to Excel",
            data=excel_data,
            file_name=f"attendance_{results['date'].replace(' ','_').replace(':','-')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5)
    confidence = st.slider("Face confidence", 0.5, 1.0, 0.7)

# ---------------- FACE DETECTION ----------------
def extract_faces(image, confidence_threshold=0.7):
    img_rgb = np.array(image.convert("RGB"))
    faces = []
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True
        )
        for face_obj in face_objs:
            if face_obj["confidence"] < confidence_threshold:
                continue
            region = face_obj["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            face = img_rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face_img = Image.fromarray(face).resize((128, 128))
            faces.append({"face": face_img, "box": (x, y, w, h)})
    except Exception as e:
        st.warning(f"Face detection error: {e}")
    return faces, img_rgb

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

# ---------------- RECOGNITION ----------------
def recognize_faces(image_pil, confidence_threshold=0.7, threshold=0.5):
    faces, original_img_rgb = extract_faces(image_pil, confidence_threshold)

    present_students = {}
    recognized_faces = []

    for i, data in enumerate(faces):
        img = data["face"]
        box = data["box"]

        emb = get_embedding(img)

        avg_distances = {}
        for name, ref_embs in reference_embeddings.items():
            avg_distances[name] = min([cosine_distance(emb, r) for r in ref_embs])

        best_name, best_dist = min(avg_distances.items(), key=lambda x: x[1])
        if best_dist > threshold:
            best_name = None

        if best_name:
            present_students[best_name] = {"img": img}
            recognized_faces.append({"name": best_name, "box": box})
        else:
            recognized_faces.append({"name": "Unknown", "box": box})

    img_draw = Image.fromarray(original_img_rgb)
    draw = ImageDraw.Draw(img_draw)

    for face in recognized_faces:
        x, y, w, h = face["box"]
        label = face["name"]
        draw.rectangle([x, y, x+w, y+h], outline=(255,0,0), width=3)
        draw.text((x, y-20), label, fill=(255,0,0))

    st.image(img_draw, use_column_width=True)

    missing = [s for s in STUDENT_ROSTER if s not in present_students]
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.session_state.last_results = {
        "present": list(present_students.keys()),
        "missing": missing,
        "date": date_str
    }

    st.write("Present:", list(present_students.keys()))
    st.write("Missing:", missing)

# ---------------- MAIN ----------------
uploaded = st.file_uploader("Upload class image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded)
    if st.button("Scan"):
        recognize_faces(img, confidence, threshold)
