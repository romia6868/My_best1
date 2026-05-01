
# streamlit_app_fixed.py
import sys
import os
import traceback
import zipfile
import random
import json
from datetime import datetime
from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd

# --- Lazy imports for heavy optional libs ---
def lazy_import_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception as e:
        print("lazy_import_deepface failed:", repr(e))
        return None

def lazy_import_rembg():
    try:
        from rembg import remove
        return remove
    except Exception as e:
        print("lazy_import_rembg failed:", repr(e))
        return None

DeepFace = lazy_import_deepface()
remove = lazy_import_rembg()

# --- Page config ---
st.set_page_config(page_title="Smart Attendance", layout="wide", initial_sidebar_state="expanded")

# --- Session defaults ---
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

# --- Constants and paths ---
ABSENCE_THRESHOLD = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIAMESE_WEIGHTS_PATH = os.path.join(BASE_DIR, "my_siamese3_weights.weights.h5")
SIAMESE_THRESHOLD = 0.49

ZIP_PATH = os.path.join(BASE_DIR, "My_Classmates_small.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "My_Classmates")
REFERENCE_DIR = os.path.join(EXTRACT_PATH, "content", "My_Classmates_small")
ROSTER_FILE = os.path.join(BASE_DIR, "student_roster.json")

# --- Utilities: roster, persistence ---
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

# --- Ensure reference dataset extracted ---
if not os.path.exists(EXTRACT_PATH) and os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

if "student_roster" not in st.session_state:
    st.session_state.student_roster = load_roster()
STUDENT_ROSTER = st.session_state.student_roster

# --- Siamese model loader (optional) ---
def load_siamese_model():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2, mobilenet_v2

        IMG_SHAPE = (128, 128, 3)

        def build_pro_embedding():
            base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
            base_model.trainable = True
            for layer in base_model.layers[:-50]:
                layer.trainable = False

            model = models.Sequential([
                layers.Lambda(mobilenet_v2.preprocess_input),
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
        print("Could not load Siamese model:", repr(e))
        return None

siamese_model = load_siamese_model()

# --- Embeddings caches (lazy) ---
@st.cache_resource
def load_reference_embeddings():
    embeddings = {}
    if DeepFace is None or not os.path.exists(REFERENCE_DIR):
        return embeddings
    for student in os.listdir(REFERENCE_DIR):
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            student_embeddings = []
            for file in os.listdir(student_path):
                if file.lower().endswith((".jpg",".jpeg",".png",".jfif")):
                    img_path = os.path.join(student_path, file)
                    try:
                        result = DeepFace.represent(img_path=img_path, model_name="Facenet512", detector_backend="retinaface", enforce_detection=False)
                        emb = np.array(result[0]["embedding"])
                        emb = emb / np.linalg.norm(emb)
                        student_embeddings.append(emb)
                    except Exception:
                        pass
            if student_embeddings:
                embeddings[student] = student_embeddings
    return embeddings

@st.cache_resource
def load_siamese_embeddings(_siamese_model):
    embeddings = {}
    if _siamese_model is None or not os.path.exists(REFERENCE_DIR):
        return embeddings
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    for student in os.listdir(REFERENCE_DIR):
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            student_embeddings = []
            for file in os.listdir(student_path):
                if file.lower().endswith((".jpg",".jpeg",".png",".jfif")):
                    img_path = os.path.join(student_path, file)
                    try:
                        img = Image.open(img_path).convert("RGB").resize((128, 128))
                        img_arr = np.array(img, dtype=np.float32)
                        img_arr = preprocess_input(img_arr)
                        img_arr = np.expand_dims(img_arr, axis=0)
                        emb = _siamese_model.predict(img_arr, verbose=0)[0]
                        student_embeddings.append(emb)
                    except Exception as e:
                        print("Error embedding:", e)
            if student_embeddings:
                embeddings[student] = student_embeddings
    return embeddings

@st.cache_resource
def load_reference_photos():
    photos = {}
    if not os.path.exists(REFERENCE_DIR):
        return photos
    for student in STUDENT_ROSTER:
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            files = [f for f in os.listdir(student_path) if f.lower().endswith((".jpg",".jpeg",".png",".jfif"))]
            if files:
                img_path = os.path.join(student_path, files[0])
                photos[student] = Image.open(img_path).convert("RGB")
    return photos

reference_embeddings = load_reference_embeddings()
siamese_embeddings = load_siamese_embeddings(siamese_model)
reference_photos = load_reference_photos()

# --- Math helpers ---
def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

# --- Embedding helper for siamese ---
def get_embedding_siamese(pil_img):
    if siamese_model is None:
        raise RuntimeError("Siamese model not loaded")
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = pil_img.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    emb = siamese_model.predict(arr, verbose=0)[0]
    return emb

# --- Robust extract_faces with debug and fallbacks ---
def extract_faces(image_pil, confidence_threshold=0.7, debug=False):
    """
    Returns (faces, original_img_rgb)
    faces: list of dict { "face": PIL.Image, "box": [x,y,w,h], "confidence": float }
    """
    original_img_rgb = np.array(image_pil.convert("RGB"))
    faces = []

    if DeepFace is None:
        if debug:
            print("extract_faces: DeepFace not available")
        return faces, original_img_rgb

    # Upscale small images to help detectors
    h, w = original_img_rgb.shape[:2]
    scale_img = image_pil
    if max(h, w) < 800:
        try:
            scale = 800 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            scale_img = image_pil.resize((new_w, new_h), Image.LANCZOS)
            if debug:
                print(f"extract_faces: upscaled image {(w,h)} -> {(new_w,new_h)}")
        except Exception as e:
            if debug:
                print("extract_faces: upscale failed", repr(e))

    backends = ["retinaface", "mtcnn", "opencv"]
    detections = []
    last_exc = None

    for backend in backends:
        try:
            if debug:
                print("extract_faces: trying backend", backend)
            detections = DeepFace.extract_faces(img_path=scale_img, detector_backend=backend, enforce_detection=False)
            if debug:
                print(f"extract_faces: backend {backend} returned {len(detections)} detections")
                if len(detections) > 0:
                    print("sample detection keys:", list(detections[0].keys()))
            if detections:
                break
        except Exception as e:
            last_exc = e
            if debug:
                print(f"extract_faces: backend {backend} exception:", repr(e))
            detections = []

    if not detections:
        if debug:
            print("extract_faces: no detections from any backend", repr(last_exc))
        return faces, original_img_rgb

    for idx, det in enumerate(detections):
        try:
            if debug:
                print(f"det[{idx}] keys:", list(det.keys()))
            conf = det.get("confidence", 0) or 0.0
            if conf < confidence_threshold:
                if debug:
                    print(f"det[{idx}] low confidence {conf}, skipping")
                continue

            face_crop = det.get("face", None)
            if face_crop is None:
                fa = det.get("facial_area", {}) or det.get("region", {}) or det.get("box", {})
                if isinstance(fa, dict) and all(k in fa for k in ("x","y","w","h")):
                    x = int(fa["x"]); y = int(fa["y"]); w_box = int(fa["w"]); h_box = int(fa["h"])
                    face_crop = original_img_rgb[y:y+h_box, x:x+w_box]
                elif isinstance(fa, dict) and all(k in fa for k in ("left","top","right","bottom")):
                    x = int(fa["left"]); y = int(fa["top"]); w_box = int(fa["right"]) - x; h_box = int(fa["bottom"]) - y
                    face_crop = original_img_rgb[y:y+h_box, x:x+w_box]
                else:
                    if debug:
                        print(f"det[{idx}] no face crop and cannot manual crop, skipping")
                    continue

            if isinstance(face_crop, np.ndarray):
                face_pil = Image.fromarray(face_crop)
            elif isinstance(face_crop, Image.Image):
                face_pil = face_crop
            else:
                try:
                    face_pil = Image.fromarray(np.array(face_crop))
                except Exception:
                    if debug:
                        print(f"det[{idx}] unknown face crop type, skipping")
                    continue

            fa = det.get("facial_area", {}) or det.get("region", {}) or det.get("box", {})
            if isinstance(fa, dict):
                if "x" in fa and "y" in fa and "w" in fa and "h" in fa:
                    box = [int(fa.get("x",0)), int(fa.get("y",0)), int(fa.get("w",0)), int(fa.get("h",0))]
                elif "left" in fa and "top" in fa and "right" in fa and "bottom" in fa:
                    x = int(fa.get("left",0)); y = int(fa.get("top",0))
                    box = [x, y, int(fa.get("right",0)) - x, int(fa.get("bottom",0)) - y]
                else:
                    box = [0,0,original_img_rgb.shape[1], original_img_rgb.shape[0]]
            elif isinstance(fa, (list, tuple)) and len(fa) >= 4:
                box = [int(fa[0]), int(fa[1]), int(fa[2]), int(fa[3])]
            else:
                box = [0,0,original_img_rgb.shape[1], original_img_rgb.shape[0]]

            faces.append({"face": face_pil, "box": box, "confidence": float(conf)})
            if debug:
                print(f"extract_faces: appended face idx={idx} box={box} conf={conf}")

        except Exception as e:
            if debug:
                print(f"extract_faces: per-detection error idx={idx}:", repr(e))
            continue

    return faces, original_img_rgb

# --- Recognition function ---
def recognize_faces(image_pil, confidence_threshold=0.7, threshold=0.4):
    use_siamese = (st.session_state.model_choice == "My Siamese Network" and siamese_model is not None)

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
    faces, original_img_rgb = extract_faces(image_pil, confidence_threshold, debug=False)
    progress.progress(30, text="Analyzing faces...")
    scan_placeholder.empty()

    if use_siamese:
        st.markdown('<div class="model-badge model-badge-siamese"><span class="material-symbols-outlined" style="font-size:14px;">check_circle</span> Using: My Siamese Network</div>', unsafe_allow_html=True)
        active_embeddings = siamese_embeddings
        active_threshold = SIAMESE_THRESHOLD
    else:
        st.markdown('<div class="model-badge model-badge-deepface"><span class="material-symbols-outlined" style="font-size:14px;">hub</span> Using: DeepFace Facenet512</div>', unsafe_allow_html=True)
        active_embeddings = reference_embeddings
        active_threshold = threshold

    present_students = {}
    recognized_faces = []
    total = max(len(faces), 1)

    for i, data in enumerate(faces):
        img = data["face"]
        box = data["box"]
        progress.progress(30 + int(60 * i / total), text=f"Identifying face {i+1} of {len(faces)}...")

        try:
            if use_siamese:
                emb = get_embedding_siamese(img)
            else:
                result = DeepFace.represent(img_path=np.array(img), model_name="Facenet512", detector_backend="skip", enforce_detection=False)
                emb = np.array(result[0]["embedding"])
                emb = emb / np.linalg.norm(emb)
        except Exception as e:
            st.write(f"❌ Failed to extract embedding: {e}")
            continue

        if not active_embeddings:
            st.write("⚠ No reference embeddings loaded!")
            continue

        distances = {}
        for name, ref_embs in active_embeddings.items():
            if use_siamese:
                d = min(euclidean_distance(emb, r) for r in ref_embs)
            else:
                d = min(cosine_distance(emb, r) for r in ref_embs)
            distances[name] = d

        best_name, best_dist = min(distances.items(), key=lambda x: x[1])

        if best_dist <= active_threshold:
            if best_name not in present_students:
                present_students[best_name] = {"img": img, "unknown": False}
                recognized_faces.append({"name": best_name, "box": box, "dist": best_dist, "unknown": False})
        else:
            unknown_key = f"Unknown_{i}"
            present_students[unknown_key] = {"img": img, "unknown": True}
            recognized_faces.append({"name": "Unknown", "box": box, "dist": best_dist, "unknown": True})

    progress.progress(100, text="Done!")
    progress.empty()

    st.markdown(f'<p style="color:#b09080;font-size:13px;margin-bottom:1rem;">{len(faces)} faces detected</p>', unsafe_allow_html=True)

    img_draw = Image.fromarray(original_img_rgb)
    draw = ImageDraw.Draw(img_draw)
    font_name = font_conf = None
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(path):
            font_name = ImageFont.truetype(path, 32)
            font_conf = ImageFont.truetype(path, 20)
            break
    if not font_name:
        font_name = ImageFont.load_default()
        font_conf = ImageFont.load_default()

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
    st.session_state.last_results = {"present": list(known_present.keys()), "missing": missing, "date": date_str}

    all_students = [d for d in os.listdir(REFERENCE_DIR) if os.path.isdir(os.path.join(REFERENCE_DIR, d))] if os.path.exists(REFERENCE_DIR) else []
    if len(known_present) == len(all_students) and len(all_students) > 0:
        st.success("🎉 Everyone is here!")
        try:
            st.audio("3.mp3", autoplay=True)
        except Exception:
            pass

    # Summary cards
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

    if any(v["unknown"] for v in present_students.values()):
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

    # Present / absent lists
    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-present"><span class="material-symbols-outlined">how_to_reg</span>Present</span><div class="divider-line"></div></div>', unsafe_allow_html=True)
    if present_students:
        cols = st.columns(5)
        for i, (name, data) in enumerate(present_students.items()):
            with cols[i % 5]:
                st.markdown('<div class="student-card">', unsafe_allow_html=True)
                st.image(data["img"], width=100)
                if data["unknown"]:
                    st.markdown('<div style="text-align:center;color:#ff8c00;font-weight:700;font-size:13px;">Unknown</div><div style="text-align:center;color:#b07040;font-size:11px;">Not in roster</div></div>', unsafe_allow_html=True)
                else:
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

# --- UI and sidebar (minimal) ---
st.title("Smart Attendance")

with st.sidebar:
    st.header("Class roster")
    for s in STUDENT_ROSTER:
        count = st.session_state.absence_counter.get(s, 0)
        st.write(f"{s} — {count}x" if count else s)

    if st.session_state.last_results is not None:
        results = st.session_state.last_results
        excel_data = export_to_excel(results["present"], results["missing"], results["date"])
        st.download_button(label="⬇ Export to Mashov", data=excel_data, file_name=f"attendance_{results['date'].replace(' ','_').replace(':','-')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.header("Recognition Model")
    siamese_available = siamese_model is not None and len(siamese_embeddings) > 0
    model_options = ["DeepFace Facenet512"]
    if siamese_available:
        model_options.append("My Siamese Network")
    chosen_model = st.radio("Choose model", model_options, key="model_choice_radio")
    st.session_state.model_choice = chosen_model

    st.markdown("---")
    st.header("Settings")
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.4)
    confidence = st.slider("Face confidence", 0.3, 1.0, 0.6)

# --- Mode tabs ---
tab_cols = st.columns(3)
tab_data = [("upload", "Upload Photo"), ("random", "Random Class"), ("camera", "Live Camera")]
for idx, (mode_key, label) in enumerate(tab_data):
    with tab_cols[idx]:
        is_active = st.session_state.mode == mode_key
        if st.button(label, key=f"tab_{mode_key}", type="primary" if is_active else "secondary"):
            st.session_state.mode = mode_key
            st.experimental_rerun()

# --- Mode content ---
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

        # Debug run (temporary) — set debug=False in production
        faces, img = extract_faces(class_image, confidence_threshold=confidence, debug=True)
        st.write(f"DEBUG: extract_faces returned {len(faces)} faces")
        for i, f in enumerate(faces):
            st.write(f"face[{i}] box={f['box']} conf={f['confidence']} type={type(f['face'])}")

        if st.button("Scan for Attendance", key="scan_upload", type="primary"):
            recognize_faces(class_image, confidence_threshold=confidence, threshold=threshold)

elif st.session_state.mode == "random":
    st.markdown('<p class="mode-desc">Generate a random class photo with students on a classroom background.</p>', unsafe_allow_html=True)
    if st.button("Generate Class Photo", key="gen_btn", type="primary"):
        with st.spinner("Generating class photo..."):
            try:
                result_img, present = generate_class_image()
                pil_image = Image.fromarray(result_img)
                st.image(pil_image, use_column_width=True)
                present_str = ", ".join(present) if present else "Nobody"
                st.markdown(f'<p style="color:#b09080;font-size:13px;margin:8px 0;">Actually present: <span style="color:#c99566;font-weight:600;">{present_str}</span></p>', unsafe_allow_html=True)
                st.markdown("---")
                recognize_faces(pil_image, confidence_threshold=confidence, threshold=threshold)
            except Exception as e:
                st.error(f"Generate failed: {e}")

elif st.session_state.mode == "camera":
    st.markdown('<p class="mode-desc">Take a photo directly from your camera.</p>', unsafe_allow_html=True)
    camera_photo = st.camera_input("")
    if camera_photo is not None:
        class_image = Image.open(camera_photo)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        if st.button("Scan for Attendance", key="scan_camera", type="primary"):
            recognize_faces(class_image, confidence_threshold=confidence, threshold=threshold)
```
