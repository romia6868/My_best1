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

st.set_page_config(
    page_title="Smart Attendance",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "mode" not in st.session_state:
    st.session_state.mode = "upload"
if "collected_photos" not in st.session_state:
    st.session_state.collected_photos = []

st.markdown("""
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
    0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 #c9956640; }
    50% { transform: scale(1.06); box-shadow: 0 0 0 8px #c9956600; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes progressFill {
    from { width: 0%; }
}
.stApp {
    background: linear-gradient(135deg, #fdf6f0 0%, #fef9f5 50%, #fdf4ea 100%) !important;
}
.main-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #c9956630;
    margin-bottom: 1.5rem;
}
.header-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #c99566, #b5784a);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    animation: pulse 3s ease-in-out infinite;
}
.header-icon .material-symbols-outlined { font-size: 28px; color: white; }
.header-title {
    font-size: 28px; font-weight: 700;
    background: linear-gradient(90deg, #b5784a, #c99566, #d4a853);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.mode-tabs { display: flex; gap: 8px; margin-bottom: 1.5rem; }
.mode-tab {
    flex: 1; padding: 11px 16px;
    border-radius: 10px;
    border: 1px solid #c9956630;
    background: #fff; color: #a07858;
    font-size: 14px; font-weight: 500;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 7px;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.2s;
    position: relative; overflow: hidden;
}
.mode-tab:hover { border-color: #c99566; transform: translateY(-2px); box-shadow: 0 4px 12px #c9956620; }
.mode-tab:active { transform: translateY(1px); }
.mode-tab .material-symbols-outlined { font-size: 18px; }
.mode-tab.active { background: linear-gradient(135deg, #c99566, #b5784a); border-color: transparent; color: white; }
.mode-tab.active .material-symbols-outlined { color: white; }
.action-btn {
    width: 100%; padding: 13px;
    border-radius: 10px;
    background: linear-gradient(135deg, #c99566, #b5784a);
    color: white; font-size: 15px; font-weight: 600;
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.2s; margin-top: 12px;
    position: relative; overflow: hidden;
}
.action-btn:hover { filter: brightness(1.08); transform: translateY(-2px); box-shadow: 0 6px 18px #c9956640; }
.action-btn:active { transform: translateY(1px); filter: brightness(0.95); }
.action-btn .material-symbols-outlined { font-size: 20px; color: white; }
.upload-zone {
    border: 1.5px dashed #c9956650;
    border-radius: 14px; padding: 2.5rem;
    text-align: center; background: #c9956610;
    margin-bottom: 1rem; transition: all 0.2s;
}
.upload-zone:hover { border-color: #c99566; background: #c9956618; }
.upload-zone .material-symbols-outlined { font-size: 44px; color: #c99566; }
.upload-text { font-size: 15px; color: #8a5a3a; margin: 8px 0 4px; font-weight: 500; }
.upload-sub { font-size: 12px; color: #b09080; }
.stat-row { display: flex; gap: 12px; margin: 1.5rem 0; }
.stat-card {
    flex: 1; background: #fff;
    border: 1px solid #c9956625;
    border-radius: 12px; padding: 16px 18px;
    transition: all 0.2s; position: relative; overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, #c9956612, transparent);
    background-size: 400px 100%;
    animation: shimmer 2.5s infinite;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px #c9956618; border-color: #c9956640; }
.stat-label {
    font-size: 11px; color: #b09080;
    text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; align-items: center; gap: 5px; margin-bottom: 6px;
}
.stat-label .material-symbols-outlined { font-size: 14px; }
.stat-val { font-size: 28px; font-weight: 700; }
.stat-sub { font-size: 11px; color: #c0a898; margin-top: 3px; }
.stat-green { color: #7a9e6a; }
.stat-red { color: #c4605a; }
.stat-gold { background: linear-gradient(90deg,#b5784a,#d4a853); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.progress-container { background: #c9956615; border-radius: 8px; height: 6px; margin: 8px 0 16px; overflow: hidden; }
.progress-bar { height: 100%; background: linear-gradient(90deg, #c99566, #d4a853); border-radius: 8px; animation: progressFill 1.5s ease-out forwards; }
.section-divider { display: flex; align-items: center; gap: 12px; margin: 1.8rem 0 1.2rem; }
.divider-line { flex: 1; height: 1px; background: #c9956625; }
.divider-badge { font-size: 12px; padding: 4px 14px; border-radius: 20px; font-weight: 600; display: flex; align-items: center; gap: 5px; }
.divider-badge .material-symbols-outlined { font-size: 15px; }
.badge-present { background: #7a9e6a20; color: #7a9e6a; }
.badge-absent { background: #c4605a20; color: #c4605a; }
.badge-unknown { background: #ff8c0020; color: #ff8c00; }
.student-card { animation: fadeInUp 0.4s ease both; text-align: center; }
.student-card:nth-child(1) { animation-delay: 0.05s; }
.student-card:nth-child(2) { animation-delay: 0.10s; }
.student-card:nth-child(3) { animation-delay: 0.15s; }
.student-card:nth-child(4) { animation-delay: 0.20s; }
.student-card:nth-child(5) { animation-delay: 0.25s; }
[data-testid="stSidebar"] { background: #fef5ee !important; border-right: 1px solid #c9956620 !important; }
.sidebar-title { font-size: 15px; font-weight: 700; color: #5a3a2a; margin-bottom: 1rem; display: flex; align-items: center; gap: 6px; }
.sidebar-title .material-symbols-outlined { font-size: 18px; color: #c99566; }
.sidebar-student {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; background: #fff;
    border-radius: 8px; margin-bottom: 6px;
    font-size: 13px; color: #7a5a4a;
    border: 1px solid #c9956618;
    transition: all 0.2s; cursor: default;
    position: relative; overflow: hidden;
}
.sidebar-student:hover { border-color: #c99566; transform: translateX(4px); box-shadow: 2px 0 8px #c9956620; }
.sidebar-student .material-symbols-outlined { font-size: 16px; color: #c99566; }
.mode-desc { color: #b09080; font-size: 14px; margin-bottom: 1rem; }
.stButton { display: none !important; }
</style>
""", unsafe_allow_html=True)

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

if "student_roster" not in st.session_state:
    st.session_state.student_roster = load_roster()

STUDENT_ROSTER = st.session_state.student_roster

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

# ---- Header ----
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

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">tune</span> Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.4)
    confidence = st.slider("Face confidence", 0.5, 1.0, 0.7)
    st.markdown("---")
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">group</span> Class roster</div>', unsafe_allow_html=True)
    for s in STUDENT_ROSTER:
        st.markdown(f'<div class="sidebar-student"><span class="material-symbols-outlined">person</span>{s}</div>', unsafe_allow_html=True)

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
                key="save_student",
                disabled=not can_save
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
                    new_embeddings = []
                    for idx in range(len(photos_collected)):
                        img_path = os.path.join(student_dir, f"{new_name}_{idx+1}.jpg")
                        try:
                            result = DeepFace.represent(img_path=img_path, model_name="Facenet512", detector_backend="retinaface", enforce_detection=False)
                            emb = np.array(result[0]["embedding"])
                            emb = emb / np.linalg.norm(emb)
                            new_embeddings.append(emb)
                        except:
                            pass
                    if new_embeddings:
                        reference_embeddings[new_name] = new_embeddings
                st.success(f"✓ {new_name} added!")
                st.rerun()

# ---- Mode tabs ----
st.markdown(f"""
<div class="mode-tabs">
    <button class="mode-tab {'active' if st.session_state.mode == 'upload' else ''}"
        onclick="fetch('/?mode=upload').then(()=>window.location.reload())">
        <span class="material-symbols-outlined">cloud_upload</span> Upload Photo
    </button>
    <button class="mode-tab {'active' if st.session_state.mode == 'random' else ''}"
        onclick="fetch('/?mode=random').then(()=>window.location.reload())">
        <span class="material-symbols-outlined">shuffle</span> Random Class
    </button>
    <button class="mode-tab {'active' if st.session_state.mode == 'camera' else ''}"
        onclick="fetch('/?mode=camera').then(()=>window.location.reload())">
        <span class="material-symbols-outlined">photo_camera</span> Live Camera
    </button>
</div>
""", unsafe_allow_html=True)

params = st.query_params
if "mode" in params:
    st.session_state.mode = params["mode"]
    st.query_params.clear()
    st.rerun()

# ---- Functions ----
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

def recognize_faces(image_pil, confidence_threshold=0.7, threshold=0.4):
    progress = st.progress(0, text="Detecting faces...")
    faces, original_img_rgb = extract_faces(image_pil, confidence_threshold)
    progress.progress(30, text="Analyzing faces...")

    present_students = {}
    recognized_faces = []
    total = max(len(faces), 1)

    for i, data in enumerate(faces):
        img = data["face"]
        box = data["box"]
        progress.progress(30 + int(60 * i / total), text=f"Identifying face {i+1} of {len(faces)}...")
        try:
            result = DeepFace.represent(
                img_path=np.array(img),
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False
            )
            emb = np.array(result[0]["embedding"])
            emb = emb / np.linalg.norm(emb)
        except:
            continue

        avg_distances = {}
        for name, ref_embs in reference_embeddings.items():
            avg_distances[name] = min([cosine_distance(emb, r) for r in ref_embs])

        best_name, best_dist = min(avg_distances.items(), key=lambda x: x[1])
        if best_dist > threshold:
            best_name = None

        if best_name and best_name not in present_students:
            present_students[best_name] = img
            recognized_faces.append({"name": best_name, "box": box, "dist": best_dist})

    progress.progress(100, text="Done!")
    progress.empty()

    st.markdown(f'<p style="color:#b09080;font-size:13px;margin-bottom:1rem;display:flex;align-items:center;gap:5px;"><span class="material-symbols-outlined" style="font-size:16px;color:#c99566;">center_focus_strong</span> {len(faces)} faces detected</p>', unsafe_allow_html=True)

    # Draw boxes
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
        pct = int((1 - face["dist"]) * 100)
        draw.rectangle([x, y, x+w, y+h], outline=(201,149,102), width=3)
        draw.text((x, y-42), face["name"], fill=(181,120,74), font=font_name)
        draw.text((x, y-20), f"{pct}%", fill=(212,168,83), font=font_conf)

    st.image(img_draw, use_column_width=True)

    missing = [s for s in STUDENT_ROSTER if s not in present_students]
    attendance_pct = int(len(present_students) / max(len(STUDENT_ROSTER), 1) * 100)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label"><span class="material-symbols-outlined" style="color:#7a9e6a;">check_circle</span>Present</div>
            <div class="stat-val stat-green">{len(present_students)}</div>
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

    # Present
    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-present"><span class="material-symbols-outlined">how_to_reg</span>Present</span><div class="divider-line"></div></div>', unsafe_allow_html=True)
    if present_students:
        cols = st.columns(5)
        for i, (name, img) in enumerate(present_students.items()):
            with cols[i % 5]:
                st.markdown('<div class="student-card">', unsafe_allow_html=True)
                st.image(img, width=100)
                st.markdown(f'<div style="text-align:center;color:#7a9e6a;font-weight:600;font-size:13px;">{name}</div></div>', unsafe_allow_html=True)

    # Absent
    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-absent"><span class="material-symbols-outlined">person_off</span>Absent</span><div class="divider-line"></div></div>', unsafe_allow_html=True)
    if missing:
        cols = st.columns(5)
        for i, name in enumerate(missing):
            with cols[i % 5]:
                st.markdown('<div class="student-card">', unsafe_allow_html=True)
                if name in reference_photos:
                    st.image(reference_photos[name], width=100)
                st.markdown(f'<div style="text-align:center;color:#c4605a;font-weight:600;font-size:13px;">{name}</div></div>', unsafe_allow_html=True)
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
    st.markdown("""
    <button class="action-btn" onclick="document.querySelectorAll('button[kind=secondary]')[0]?.click()">
        <span class="material-symbols-outlined">face_retouching_natural</span> Scan for Attendance
    </button>
    """, unsafe_allow_html=True)
    if class_file is not None:
        class_image = Image.open(class_file)
        class_image = ImageOps.exif_transpose(class_image)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        if st.button("Scan", key="scan_upload"):
            recognize_faces(class_image, confidence, threshold)

elif st.session_state.mode == "random":
    st.markdown('<p class="mode-desc">Generate a random class photo with students on a classroom background.</p>', unsafe_allow_html=True)
    st.markdown("""
    <button class="action-btn" onclick="document.querySelectorAll('button[kind=secondary]')[0]?.click()">
        <span class="material-symbols-outlined">shuffle</span> Generate Class Photo
    </button>
    """, unsafe_allow_html=True)
    if st.button("Generate", key="gen_btn"):
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
        st.markdown("""
        <button class="action-btn" onclick="document.querySelectorAll('button[kind=secondary]')[0]?.click()">
            <span class="material-symbols-outlined">face_retouching_natural</span> Scan for Attendance
        </button>
        """, unsafe_allow_html=True)
        if st.button("Scan", key="scan_camera"):
            recognize_faces(class_image, confidence, threshold)
