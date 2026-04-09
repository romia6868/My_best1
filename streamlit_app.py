import sys
import os

try:
    import cv2
    print("cv2 OK:", cv2.__version__)
except Exception as e:
    print("cv2 FULL ERROR:", repr(e))
    import traceback
    traceback.print_exc()

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

st.set_page_config(
    page_title="Smart Attendance",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "mode" not in st.session_state:
    st.session_state.mode = "upload"
if "collected_photos" not in st.session_state:
    st.session_state.collected_photos = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "absence_counter" not in st.session_state:
    st.session_state.absence_counter = {}
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "DeepFace Facenet512"

ABSENCE_THRESHOLD = 3
SIAMESE_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_siamese3_weights.weights.h5")
SIAMESE_THRESHOLD = 0.49  # Best Threshold (95% Recall)

css = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap"/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0"/>
<style>
* { font-family: 'Space Grotesk', sans-serif !important; }
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal; font-style: normal; font-size: 22px;
    line-height: 1; letter-spacing: normal; text-transform: none;
    display: inline-block; white-space: nowrap;
    -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 #b8a9c940; }
    50% { transform: scale(1.06); box-shadow: 0 0 0 8px #b8a9c900; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes progressFill { from { width: 0%; } }
@keyframes scanLine {
    0% { top: 0%; opacity: 1; }
    100% { top: 100%; opacity: 0.3; }
}
.stApp { background: #f0eef4 !important; }
.main-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #e4dff0;
    margin-bottom: 1.5rem;
}
.header-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #b8a9c9, #9585b0);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    animation: pulse 3s ease-in-out infinite;
}
.header-icon .material-symbols-outlined { font-size: 28px; color: white; }
.header-title {
    font-size: 28px; font-weight: 700;
    background: linear-gradient(90deg, #6b5a8a, #9585b0, #c4b8d8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.scan-container { position: relative; display: inline-block; width: 100%; }
.scan-overlay {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none; z-index: 10; border-radius: 8px; overflow: hidden;
}
.scan-line {
    position: absolute; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #b8a9c9, #c4b8d8, #b8a9c9, transparent);
    animation: scanLine 1.5s ease-in-out infinite;
    box-shadow: 0 0 12px #b8a9c980;
}
.upload-zone {
    border: 1.5px dashed #c4b8d8;
    border-radius: 14px; padding: 2.5rem;
    text-align: center; background: #ebe8f240;
    margin-bottom: 1rem; transition: all 0.2s;
}
.upload-zone:hover { border-color: #9585b0; background: #ebe8f260; }
.upload-zone .material-symbols-outlined { font-size: 44px; color: #9585b0; }
.upload-text { font-size: 15px; color: #4a3a6a; margin: 8px 0 4px; font-weight: 500; }
.upload-sub { font-size: 12px; color: #a098b8; }
.stat-row { display: flex; gap: 12px; margin: 1.5rem 0; }
.stat-card {
    flex: 1; background: #fff;
    border: 1px solid #e4dff0;
    border-radius: 12px; padding: 16px 18px;
    transition: all 0.2s; position: relative; overflow: hidden;
}
.stat-card::after {
    content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, #ebe8f230, transparent);
    background-size: 400px 100%; animation: shimmer 2.5s infinite;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px #b8a9c920; border-color: #c4b8d8; }
.stat-label {
    font-size: 11px; color: #a098b8; text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; align-items: center; gap: 5px; margin-bottom: 6px;
}
.stat-label .material-symbols-outlined { font-size: 14px; }
.stat-val { font-size: 28px; font-weight: 700; }
.stat-sub { font-size: 11px; color: #c4b8d8; margin-top: 3px; }
.stat-green { color: #68b88a; }
.stat-red { color: #d4707a; }
.stat-gold { background: linear-gradient(90deg,#6b5a8a,#b8a9c9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.progress-container { background: #ebe8f2; border-radius: 8px; height: 6px; margin: 8px 0 16px; overflow: hidden; }
.progress-bar { height: 100%; background: linear-gradient(90deg, #b8a9c9, #c4b8d8); border-radius: 8px; animation: progressFill 1.5s ease-out forwards; }
.section-divider { display: flex; align-items: center; gap: 12px; margin: 1.8rem 0 1.2rem; }
.divider-line { flex: 1; height: 1px; background: #e4dff0; }
.divider-badge { font-size: 12px; padding: 4px 14px; border-radius: 20px; font-weight: 600; display: flex; align-items: center; gap: 5px; }
.divider-badge .material-symbols-outlined { font-size: 15px; }
.badge-present { background: #68b88a20; color: #68b88a; }
.badge-absent { background: #d4707a20; color: #d4707a; }
.badge-unknown { background: #e8a85020; color: #e8a850; }
.student-card { animation: fadeInUp 0.4s ease both; text-align: center; }
.student-card:nth-child(1) { animation-delay: 0.05s; }
.student-card:nth-child(2) { animation-delay: 0.10s; }
.student-card:nth-child(3) { animation-delay: 0.15s; }
.student-card:nth-child(4) { animation-delay: 0.20s; }
.student-card:nth-child(5) { animation-delay: 0.25s; }
[data-testid="stSidebar"] { background: #e8e4f0 !important; border-right: 1px solid #e4dff0 !important; }
.sidebar-title { font-size: 15px; font-weight: 700; color: #4a3a6a; margin-bottom: 1rem; display: flex; align-items: center; gap: 6px; }
.sidebar-title .material-symbols-outlined { font-size: 18px; color: #9585b0; }
.sidebar-student {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; background: #f0eef4;
    border-radius: 8px; margin-bottom: 6px;
    font-size: 13px; color: #4a3a6a;
    border: 1px solid #e4dff0;
    transition: all 0.2s; cursor: default;
}
.sidebar-student:hover { border-color: #b8a9c9; transform: translateX(4px); box-shadow: 2px 0 8px #b8a9c920; }
.sidebar-student .material-symbols-outlined { font-size: 16px; color: #9585b0; }
.mode-desc { color: #a098b8; font-size: 14px; margin-bottom: 1rem; }
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
    margin-bottom: 1rem;
}
.model-badge-deepface { background: #9585b020; color: #9585b0; border: 1px solid #9585b040; }
.model-badge-siamese { background: #68b88a20; color: #68b88a; border: 1px solid #68b88a40; }
</style>
"""

button_css = """
<style>
.stButton > button {
    background: #ebe8f2 !important; color: #4a3a6a !important;
    border: 1.5px solid #e4dff0 !important; border-radius: 10px !important;
    padding: 11px 16px !important; font-size: 14px !important;
    font-weight: 500 !important; width: 100% !important;
    transition: all 0.2s !important;
    font-family: 'Space Grotesk', sans-serif !important; margin-top: 0 !important;
}
.stButton > button:hover {
    border-color: #9585b0 !important; transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px #b8a9c930 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #b8a9c9, #9585b0) !important;
    color: white !important; border: none !important;
    box-shadow: 0 4px 14px #b8a9c940 !important;
    padding: 13px 28px !important; font-size: 15px !important;
    font-weight: 600 !important; margin-top: 12px !important;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08) !important; transform: translateY(-2px) !important;
}
.stDownloadButton > button {
    background: #ebe8f2 !important; color: #9585b0 !important;
    border: 1.5px solid #b8a9c9 !important; border-radius: 10px !important;
    font-size: 13px !important; font-weight: 600 !important;
    width: 100% !important; transition: all 0.2s !important; margin-top: 8px !important;
}
.stDownloadButton > button:hover { background: #e4dff0 !important; transform: translateY(-1px) !important; }
</style>
"""

st.markdown(css, unsafe_allow_html=True)
st.markdown(button_css, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "My_Classmates_small.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "My_Classmates")
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = os.path.join(EXTRACT_PATH, "content", "My_Classmates_small")
ROSTER_FILE = os.path.join(BASE_DIR, "student_roster.json")

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


def load_siamese_model():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2, mobilenet_v2

        IMG_SHAPE = (128, 128, 3)

        def build_pro_embedding():
            base_model = MobileNetV2(
                input_shape=IMG_SHAPE,
                include_top=False,
                weights='imagenet'
            )

            base_model.trainable = True
            for layer in base_model.layers[:-50]:
                layer.trainable = False

            model = models.Sequential([
                layers.Lambda(mobilenet_v2.preprocess_input),   # ❗ חובה
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.Dense(128, activation=None),
                layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2_norm")
            ], name="MobileNetV2_Embedding")

            return model

        embedding_model = build_pro_embedding()

        dummy = tf.zeros((1, 128, 128, 3))
        _ = embedding_model(dummy)

        embedding_model.load_weights(SIAMESE_WEIGHTS_PATH)

        return embedding_model

    except Exception as e:
        st.error(f"Could not load Siamese model: {e}")
        return None


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
                        result = DeepFace.represent(
                            img_path=img_path,
                            model_name="Facenet512",
                            detector_backend="retinaface",
                            enforce_detection=False
                        )
                        emb = np.array(result[0]["embedding"])
                        emb = emb / np.linalg.norm(emb)
                        student_embeddings.append(emb)
                    except:
                        pass
            if student_embeddings:
                embeddings[student] = student_embeddings
    return embeddings

@st.cache_resource
def load_siamese_embeddings(_siamese_model):
    """בונה embeddings של כל התלמידים עם הרשת הסיאמית"""
    if _siamese_model is None:
        return {}
    embeddings = {}
    for student in os.listdir(REFERENCE_DIR):
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            student_embeddings = []
            for file in os.listdir(student_path):
                if file.lower().endswith((".jpg",".jpeg",".png",".jfif")):
                    img_path = os.path.join(student_path, file)
                    try:
                        img = Image.open(img_path).convert("RGB").resize((128, 128))
                        img_arr = np.array(img, dtype=np.float32) / 255.0
                        img_arr = np.expand_dims(img_arr, axis=0)
                        emb = _siamese_model.predict(img_arr, verbose=0)[0]
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

siamese_model = load_siamese_model()
reference_embeddings = load_reference_embeddings()
siamese_embeddings = load_siamese_embeddings(siamese_model)
reference_photos = load_reference_photos()

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

with st.sidebar:
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">group</span> Class roster</div>', unsafe_allow_html=True)
    for s in STUDENT_ROSTER:
        count = st.session_state.absence_counter.get(s, 0)
        if count >= ABSENCE_THRESHOLD:
            badge = f'<span style="margin-left:auto;background:#c4605a;color:white;font-size:11px;font-weight:700;padding:2px 7px;border-radius:10px;">!{count}</span>'
        elif count > 0:
            badge = f'<span style="margin-left:auto;color:#b09080;font-size:11px;">{count}x</span>'
        else:
            badge = ''
        st.markdown(f'<div class="sidebar-student"><span class="material-symbols-outlined">person</span>{s}{badge}</div>', unsafe_allow_html=True)

    if st.session_state.last_results is not None:
        results = st.session_state.last_results
        excel_data = export_to_excel(results["present"], results["missing"], results["date"])
        st.download_button(
            label="⬇ Export to Mashov",
            data=excel_data,
            file_name=f"attendance_{results['date'].replace(' ','_').replace(':','-')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="export_btn"
        )
    else:
        st.markdown('<p style="color:#c0a898;font-size:12px;margin-top:8px;">Run a scan to enable export</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ---- בחירת מודל ----
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">model_training</span> Recognition Model</div>', unsafe_allow_html=True)

    siamese_available = siamese_model is not None and len(siamese_embeddings) > 0
    model_options = ["DeepFace Facenet512"]
    if siamese_available:
        model_options.append("My Siamese Network")

    chosen_model = st.radio(
        "Choose model",
        model_options,
        key="model_choice_radio",
        help="Siamese = custom-trained on your classmates. Facenet512 = pretrained general model."
    )
    st.session_state.model_choice = chosen_model

    if chosen_model == "My Siamese Network":
        st.markdown('<div class="model-badge model-badge-siamese"><span class="material-symbols-outlined" style="font-size:14px;">check_circle</span> Custom model loaded</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#a098b8;font-size:11px;">Threshold: {SIAMESE_THRESHOLD} · Euclidean distance · 100% recall</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-badge model-badge-deepface"><span class="material-symbols-outlined" style="font-size:14px;">hub</span> Facenet512 active</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a098b8;font-size:11px;">Pretrained · Cosine distance · 512-dim</p>', unsafe_allow_html=True)

    if not siamese_available:
        st.markdown('<p style="color:#d4707a;font-size:11px;">⚠ Siamese weights not found</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">manage_accounts</span> Manage Students</div>', unsafe_allow_html=True)

    with st.expander("Remove student"):
        if STUDENT_ROSTER:
            student_to_remove = st.selectbox("Select student", STUDENT_ROSTER, key="remove_select")
            if st.button("Remove", key="remove_btn"):
                st.session_state.student_roster.remove(student_to_remove)
                save_roster(st.session_state.student_roster)
                student_path = os.path.join(REFERENCE_DIR, student_to_remove)
                if os.path.exists(student_path):
                    import shutil
                    shutil.rmtree(student_path)
                st.success(f"{student_to_remove} removed!")
                st.rerun()

    with st.expander("Add new student"):
        new_name = st.text_input("Student name", placeholder="e.g. Noa", key="new_name")
        photo_method = st.radio("Photo method", ["📷 Camera", "📤 Upload"], key="photo_method", horizontal=True)
        if new_name:
            photos_collected = []
            if photo_method == "📷 Camera":
                st.markdown(f'<p style="color:#b09080;font-size:12px;">Collected: <b style="color:#c99566;">{len(st.session_state.collected_photos)}/10</b></p>', unsafe_allow_html=True)
                if len(st.session_state.collected_photos) > 0:
                    pct = len(st.session_state.collected_photos) * 10
                    st.markdown(f'<div class="progress-container"><div class="progress-bar" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
                cam_img = st.camera_input("", key=f"cam_{len(st.session_state.collected_photos)}")
                col1, col2 = st.columns(2)
                with col1:
                    if cam_img and st.button("Add photo", key="add_photo"):
                        st.session_state.collected_photos.append(cam_img)
                        st.rerun()
                with col2:
                    if st.button("Clear all", key="clear_photos"):
                        st.session_state.collected_photos = []
                        st.rerun()
                photos_collected = st.session_state.collected_photos
            else:
                uploaded_files = st.file_uploader("Upload photos", type=["jpg","jpeg","png"], accept_multiple_files=True, key="upload_photos")
                if uploaded_files:
                    pct = min(len(uploaded_files) * 10, 100)
                    st.markdown(f'<div class="progress-container"><div class="progress-bar" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
                    st.markdown(f'<p style="color:#{"7a9e6a" if len(uploaded_files)>=5 else "c99566"};font-size:12px;">{len(uploaded_files)}/10 photos</p>', unsafe_allow_html=True)
                photos_collected = uploaded_files or []

            can_save = len(photos_collected) >= 5
            if st.button(
                "Save student" if can_save else f"Need {max(0, 5-len(photos_collected))} more",
                key="save_student", disabled=not can_save
            ):
                student_dir = os.path.join(REFERENCE_DIR, new_name)
                os.makedirs(student_dir, exist_ok=True)
                for idx, photo in enumerate(photos_collected):
                    img = Image.open(photo).convert("RGB")
                    img.save(os.path.join(student_dir, f"{new_name}_{idx+1}.jpg"))
                if new_name not in st.session_state.student_roster:
                    st.session_state.student_roster.append(new_name)
                    save_roster(st.session_state.student_roster)
                st.session_state.collected_photos = []
                with st.spinner(f"Processing {new_name}'s photos..."):
                    new_embs_deepface = []
                    new_embs_siamese = []
                    for idx in range(len(photos_collected)):
                        img_path = os.path.join(student_dir, f"{new_name}_{idx+1}.jpg")
                        # Facenet
                        try:
                            result = DeepFace.represent(img_path=img_path, model_name="Facenet512", detector_backend="retinaface", enforce_detection=False)
                            emb = np.array(result[0]["embedding"])
                            emb = emb / np.linalg.norm(emb)
                            new_embs_deepface.append(emb)
                        except:
                            pass
                        # Siamese
                        if siamese_model is not None:
                            try:
                                img = Image.open(img_path).convert("RGB").resize((128, 128))
                                img_arr = np.array(img, dtype=np.float32) / 255.0
                                img_arr = np.expand_dims(img_arr, axis=0)
                                emb = siamese_model.predict(img_arr, verbose=0)[0]
                                new_embs_siamese.append(emb)
                            except:
                                pass
                    if new_embs_deepface:
                        reference_embeddings[new_name] = new_embs_deepface
                    if new_embs_siamese:
                        siamese_embeddings[new_name] = new_embs_siamese
                st.success(f"✓ {new_name} added!")
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">tune</span> Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.4)
    confidence = st.slider("Face confidence", 0.5, 1.0, 0.7)

# ---- Mode tabs ----
tab_cols = st.columns(3)
tab_data = [("upload", "Upload Photo"), ("random", "Random Class"), ("camera", "Live Camera")]
for idx, (mode_key, label) in enumerate(tab_data):
    with tab_cols[idx]:
        is_active = st.session_state.mode == mode_key
        if st.button(label, key=f"tab_{mode_key}", type="primary" if is_active else "secondary"):
            st.session_state.mode = mode_key
            st.rerun()

def generate_class_image():
    background_options = [
        os.path.join(BASE_DIR, "הורדה.jfif"),
        os.path.join(BASE_DIR, "images (1).jfif"),
        os.path.join(BASE_DIR, "images.jfif"),
        os.path.join(BASE_DIR, "images (2).jfif"),
    ]
    available_backgrounds = [b for b in background_options if os.path.exists(b)]
    if not available_backgrounds:
        st.error("No background images found")
        st.stop()
    bg = cv2.imread(random.choice(available_backgrounds))
    if bg is None:
        st.error("Could not load background")
        st.stop()
    bg = cv2.resize(bg, (900, 600), interpolation=cv2.INTER_CUBIC)
    students = os.listdir(REFERENCE_DIR)
    present = random.sample(students, random.randint(0, len(students)))
    rows, cols = 2, 5
    cell_w = bg.shape[1] // cols
    cell_h = bg.shape[0] // rows
    positions = [(c * cell_w, r * cell_h) for r in range(rows) for c in range(cols)]
    random.shuffle(positions)
    bg_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")
    i = 0
    for name in present:
        if i < len(positions):
            student_dir = os.path.join(REFERENCE_DIR, name)
            imgs = os.listdir(student_dir)
            if imgs:
                face = cv2.imread(os.path.join(student_dir, random.choice(imgs)))
                if face is not None:
                    new_w = int(cell_w * 0.8)
                    new_h = int(cell_h * 0.8)
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_no_bg = remove(face_pil).resize((new_w, new_h))
                    x, y = positions[i]
                    x += (cell_w - new_w) // 2
                    y += (cell_h - new_h) // 2
                    bg_pil.paste(face_no_bg, (x, y), face_no_bg)
                    i += 1
    return np.array(bg_pil.convert("RGB")), present

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
            pad_x = int(0.2 * w)
            pad_y = int(0.2 * h)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_rgb.shape[1], x + w + pad_x)
            y2 = min(img_rgb.shape[0], y + h + pad_y)
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_img = Image.fromarray(face).resize((160, 160))
            faces.append({"face": face_img, "box": (x1, y1, x2-x1, y2-y1)})
    except Exception as e:
        st.warning(f"Face detection error: {e}")
    return faces, img_rgb

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_embedding_siamese(face_img):
    img = face_img.convert("RGB").resize((128, 128))
    img_arr = np.array(img, dtype=np.float32)
    img_arr = preprocess_input(img_arr)   # ❗ חובה
    img_arr = np.expand_dims(img_arr, axis=0)
    emb = siamese_model.predict(img_arr, verbose=0)[0]
    return emb


def recognize_faces(image_pil, confidence_threshold=0.7, threshold=0.4):
    use_siamese = (
        st.session_state.model_choice == "My Siamese Network"
        and siamese_model is not None
    )

    # --- UI: סריקה ---
    scan_placeholder = st.empty()
    scan_placeholder.markdown("""
    <div class="scan-container">
        <div class="scan-overlay"><div class="scan-line"></div></div>
        <div style="background:#c9956615;border-radius:8px;padding:2rem;text-align:center;">
            <span class="material-symbols-outlined" style="font-size:48px;color:#c99566;">document_scanner</span>
            <p style="color:#b09080;margin-top:8px;font-size:14px;">Scanning photo...</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    progress = st.progress(0, text="Detecting faces...")
    faces, original_img_rgb = extract_faces(image_pil, confidence_threshold)
    progress.progress(30, text="Analyzing faces...")
    scan_placeholder.empty()

    # --- בחירת מודל ---
    if use_siamese:
        st.markdown(
            '<div class="model-badge model-badge-siamese"><span class="material-symbols-outlined" style="font-size:14px;">check_circle</span> Using: My Siamese Network</div>',
            unsafe_allow_html=True,
        )
        active_embeddings = siamese_embeddings
        active_threshold = SIAMESE_THRESHOLD
    else:
        st.markdown(
            '<div class="model-badge model-badge-deepface"><span class="material-symbols-outlined" style="font-size:14px;">hub</span> Using: DeepFace Facenet512</div>',
            unsafe_allow_html=True,
        )
        active_embeddings = reference_embeddings
        active_threshold = threshold

    # --- לולאת השוואה חדשה ---
    present_students = {}
    recognized_faces = []
    total = max(len(faces), 1)

    for i, data in enumerate(faces):
        img = data["face"]
        box = data["box"]
        progress.progress(
            30 + int(60 * i / total),
            text=f"Identifying face {i+1} of {len(faces)}...",
        )

        # --- הפקת embedding ---
        try:
            if use_siamese:
                emb = get_embedding_siamese(img)
            else:
                result = DeepFace.represent(
                    img_path=np.array(img),
                    model_name="Facenet512",
                    detector_backend="skip",
                    enforce_detection=False,
                )
                emb = np.array(result[0]["embedding"])
                emb = emb / np.linalg.norm(emb)
        except Exception as e:
            st.write(f"❌ Failed to extract embedding: {e}")
            continue

        # --- אם אין אמבדינגים להשוואה ---
        if not active_embeddings:
            st.write("⚠ No reference embeddings loaded!")
            continue

        # --- חישוב מרחקים ---
        distances = {}
        for name, ref_embs in active_embeddings.items():
            if use_siamese:
                d = min(euclidean_distance(emb, r) for r in ref_embs)
            else:
                d = min(cosine_distance(emb, r) for r in ref_embs)

            distances[name] = d
            st.write(f"🔍 Comparing to {name}: distance = {d}")

        # --- בחירת ההתאמה הטובה ביותר ---
        best_name, best_dist = min(distances.items(), key=lambda x: x[1])
        st.write(f"➡ Best match: {best_name} (dist={best_dist}, threshold={active_threshold})")

        # --- החלטה אם זה ידוע או לא ---
        if best_dist <= active_threshold:
            if best_name not in present_students:
                present_students[best_name] = {"img": img, "unknown": False}
                recognized_faces.append(
                    {
                        "name": best_name,
                        "box": box,
                        "dist": best_dist,
                        "unknown": False,
                    }
                )
        else:
            unknown_key = f"Unknown_{i}"
            present_students[unknown_key] = {"img": img, "unknown": True}
            recognized_faces.append(
                {
                    "name": "Unknown",
                    "box": box,
                    "dist": best_dist,
                    "unknown": True,
                }
            )

    # --- המשך הפונקציה (ציור, סטטיסטיקות וכו') ---
    progress.progress(100, text="Done!")
    progress.empty()

    # כל שאר הקוד שלך נשאר בדיוק אותו דבר
    # (ציור תיבות, חישוב נוכחות, UI וכו')


    st.markdown(f'<p style="color:#b09080;font-size:13px;margin-bottom:1rem;">{len(faces)} faces detected</p>', unsafe_allow_html=True)

    img_draw = Image.fromarray(original_img_rgb)
    draw = ImageDraw.Draw(img_draw)
    font_name = font_conf = None
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(path):
            font_name = ImageFont.truetype(path, 32)
            font_conf = ImageFont.truetype(path, 20)
            break
    if not font_name:
        font_name = ImageFont.load_default(size=32)
        font_conf = ImageFont.load_default(size=20)

    for face in recognized_faces:
        x, y, w, h = face["box"]
        if face["unknown"]:
            draw.rectangle([x, y, x+w, y+h], outline=(220,100,30), width=3)
            draw.text((x, y-42), "Unknown", fill=(220,100,30), font=font_name)
        else:
            pct = int((1 - face["dist"]) * 100) if not use_siamese else int(max(0, (1 - face["dist"] / active_threshold)) * 100)
            draw.rectangle([x, y, x+w, y+h], outline=(201,149,102), width=3)
            draw.text((x, y-42), face["name"], fill=(181,120,74), font=font_name)
            draw.text((x, y-20), f"{pct}%", fill=(212,168,83), font=font_conf)

    st.image(img_draw, use_column_width=True)

    known_present = {k: v for k, v in present_students.items() if not v["unknown"]}
    missing = [s for s in STUDENT_ROSTER if s not in known_present]
    attendance_pct = int(len(known_present) / max(len(STUDENT_ROSTER), 1) * 100)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    updated_absences = update_absences(missing)
    st.session_state.last_results = {
        "present": list(known_present.keys()),
        "missing": missing,
        "date": date_str
    }

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label"><span class="material-symbols-outlined" style="color:#7a9e6a;">check_circle</span>Present</div>
            <div class="stat-val stat-green">{len(known_present)}</div>
            <div class="stat-sub">out of {len(STUDENT_ROSTER)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label"><span class="material-symbols-outlined" style="color:#c4605a;">cancel</span>Absent</div>
            <div class="stat-val stat-red">{len(missing)}</div>
            <div class="stat-sub">check required</div>
        </div>
        <div class="stat-card">
            <div class="stat-label"><span class="material-symbols-outlined" style="color:#d4a853;">insights</span>Attendance</div>
            <div class="stat-val stat-gold">{attendance_pct}%</div>
            <div class="stat-sub">today</div>
        </div>
    </div>
    <div class="progress-container">
        <div class="progress-bar" style="width:{attendance_pct}%"></div>
    </div>
    """, unsafe_allow_html=True)

    chronic_absent = [s for s in missing if updated_absences.get(s, 0) >= ABSENCE_THRESHOLD]
    if chronic_absent:
        names = ", ".join(chronic_absent)
        st.markdown(f"""
        <div style="background:#c4605a15;border:1.5px solid #c4605a50;border-radius:12px;
            padding:14px 18px;margin-bottom:1rem;display:flex;align-items:center;gap:10px;">
            <span class="material-symbols-outlined" style="color:#c4605a;font-size:24px;">notification_important</span>
            <div>
                <div style="font-weight:700;color:#a03030;font-size:14px;">Chronic absence alert!</div>
                <div style="color:#904040;font-size:12px;">{names} have been absent {ABSENCE_THRESHOLD}+ times.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    has_unknown = any(v["unknown"] for v in present_students.values())
    if has_unknown:
        st.markdown("""
        <div style="background:#ff8c0015;border:1.5px solid #ff8c0050;border-radius:12px;
            padding:14px 18px;margin-bottom:1rem;display:flex;align-items:center;gap:10px;">
            <span class="material-symbols-outlined" style="color:#ff8c00;font-size:24px;">warning</span>
            <div>
                <div style="font-weight:700;color:#c45a00;font-size:14px;">Unidentified person detected!</div>
                <div style="color:#b07040;font-size:12px;">Someone in the photo is not in the class roster.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-present"><span class="material-symbols-outlined">how_to_reg</span>Present</span><div class="divider-line"></div></div>', unsafe_allow_html=True)
    if present_students:
        cols = st.columns(5)
        for i, (name, data) in enumerate(present_students.items()):
            with cols[i % 5]:
                if data["unknown"]:
                    st.markdown('<div class="student-card">', unsafe_allow_html=True)
                    st.image(data["img"], width=100)
                    st.markdown('<div style="text-align:center;color:#ff8c00;font-weight:700;font-size:13px;">Unknown</div><div style="text-align:center;color:#b07040;font-size:11px;">Not in roster</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="student-card">', unsafe_allow_html=True)
                    st.image(data["img"], width=100)
                    st.markdown(f'<div style="text-align:center;color:#7a9e6a;font-weight:600;font-size:13px;">{name}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-absent"><span class="material-symbols-outlined">person_off</span>Absent</span><div class="divider-line"></div></div>', unsafe_allow_html=True)
    if missing:
        cols = st.columns(5)
        for i, name in enumerate(missing):
            with cols[i % 5]:
                st.markdown('<div class="student-card">', unsafe_allow_html=True)
                if name in reference_photos:
                    st.image(reference_photos[name], width=100)
                absence_count = updated_absences.get(name, 0)
                color = "#a03030" if absence_count >= ABSENCE_THRESHOLD else "#c4605a"
                badge = f'<span style="font-size:10px;background:#c4605a20;padding:2px 6px;border-radius:10px;">{absence_count}x</span>' if absence_count > 0 else ''
                st.markdown(f'<div style="text-align:center;color:{color};font-weight:600;font-size:13px;">{name} {badge}</div></div>', unsafe_allow_html=True)
    else:
        st.success("Everyone's here today!")

# ---- Mode content ----
if st.session_state.mode == "upload":
    st.markdown("""
    <div class="upload-zone">
        <span class="material-symbols-outlined">cloud_upload</span>
        <div class="upload-text">Drop your class photo here</div>
        <div class="upload-sub">JPG · PNG · JPEG</div>
    </div>
    """, unsafe_allow_html=True)
    class_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if class_file is not None:
        class_image = Image.open(class_file)
        class_image = ImageOps.exif_transpose(class_image)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        if st.button("Scan for Attendance", key="scan_upload", type="primary"):
            recognize_faces(class_image, confidence, threshold)

elif st.session_state.mode == "random":
    st.markdown('<p class="mode-desc">Generate a random class photo with students on a classroom background.</p>', unsafe_allow_html=True)
    if st.button("Generate Class Photo", key="gen_btn", type="primary"):
        with st.spinner("Generating class photo..."):
            result_img, present = generate_class_image()
        pil_image = Image.fromarray(result_img)
        st.image(pil_image, use_column_width=True)
        present_str = ", ".join(present) if present else "Nobody"
        st.markdown(f'<p style="color:#b09080;font-size:13px;margin:8px 0;">Actually present: <span style="color:#c99566;font-weight:600;">{present_str}</span></p>', unsafe_allow_html=True)
        st.markdown("---")
        recognize_faces(pil_image, confidence, threshold)

elif st.session_state.mode == "camera":
    st.markdown('<p class="mode-desc">Take a photo directly from your camera.</p>', unsafe_allow_html=True)
    camera_photo = st.camera_input("")
    if camera_photo is not None:
        class_image = Image.open(camera_photo)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        if st.button("Scan for Attendance", key="scan_camera", type="primary"):
            recognize_faces(class_image, confidence, threshold)
