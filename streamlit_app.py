import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import os
import zipfile
import random
import cv2
from rembg import remove

st.set_page_config(
    page_title="Smart Attendance",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

* { font-family: 'Space Grotesk', sans-serif !important; }
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal; font-style: normal;
    font-size: 22px; line-height: 1;
    letter-spacing: normal; text-transform: none;
    display: inline-block; white-space: nowrap;
    word-wrap: normal; direction: ltr;
    -webkit-font-feature-settings: 'liga';
    font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}
.icon-rose { color: #c99566; }
.icon-green { color: #7a9e6a; }
.icon-red { color: #c4605a; }
.icon-gold { color: #d4a853; }
.icon-muted { color: #b09080; }

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
}
.header-icon .material-symbols-outlined { font-size: 28px; color: white; }
.header-title {
    font-size: 28px; font-weight: 700;
    background: linear-gradient(90deg, #b5784a, #c99566, #d4a853);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}

.stat-row { display: flex; gap: 12px; margin: 1.5rem 0; }
.stat-card {
    flex: 1; background: #fff;
    border: 1px solid #c9956625;
    border-radius: 12px; padding: 16px 18px;
    display: flex; flex-direction: column; gap: 4px;
}
.stat-label {
    font-size: 11px; color: #b09080;
    text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; align-items: center; gap: 5px;
}
.stat-val { font-size: 28px; font-weight: 700; }
.stat-sub { font-size: 11px; color: #c0a898; }
.stat-green { color: #7a9e6a; }
.stat-red { color: #c4605a; }
.stat-gold { background: linear-gradient(90deg, #b5784a, #d4a853);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.stRadio > div { flex-direction: row !important; gap: 8px; }
.stRadio label {
    background: #fff !important;
    border: 1px solid #c9956630 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    color: #a07858 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
.stRadio label:has(input:checked) {
    background: linear-gradient(135deg, #c99566, #b5784a) !important;
    border-color: transparent !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(135deg, #c99566, #b5784a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
}

.upload-zone {
    border: 1.5px dashed #c9956650;
    border-radius: 14px; padding: 2.5rem;
    text-align: center; background: #c9956610;
    margin-bottom: 1rem;
}
.upload-icon .material-symbols-outlined { font-size: 44px; color: #c99566; }
.upload-text { font-size: 15px; color: #8a5a3a; margin: 8px 0 4px; font-weight: 500; }
.upload-sub { font-size: 12px; color: #b09080; }

.section-divider {
    display: flex; align-items: center; gap: 12px;
    margin: 1.8rem 0 1.2rem;
}
.divider-line { flex: 1; height: 1px; background: #c9956625; }
.divider-badge {
    font-size: 12px; padding: 4px 14px;
    border-radius: 20px; font-weight: 600;
    display: flex; align-items: center; gap: 5px;
}
.badge-present { background: #7a9e6a20; color: #7a9e6a; }
.badge-absent { background: #c4605a20; color: #c4605a; }
.badge-present .material-symbols-outlined,
.badge-absent .material-symbols-outlined { font-size: 15px; }

[data-testid="stSidebar"] {
    background: #fef5ee !important;
    border-right: 1px solid #c9956620 !important;
}
.sidebar-title {
    font-size: 15px; font-weight: 700; color: #5a3a2a;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 6px;
}
.sidebar-title .material-symbols-outlined { font-size: 18px; color: #c99566; }
.sidebar-student {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; background: #fff;
    border-radius: 8px; margin-bottom: 6px;
    font-size: 13px; color: #7a5a4a;
    border: 1px solid #c9956618;
}
.sidebar-student .material-symbols-outlined { font-size: 16px; color: #c99566; }

.mode-desc { color: #b09080; font-size: 14px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "My_Classmates_small.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "My_Classmates")
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = os.path.join(EXTRACT_PATH, "content", "My_Classmates_small")
STUDENT_ROSTER = ['Maayan','Tomer','Roei','Zohar','Ilay']

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

# -------------------------
# Header
# -------------------------
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

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">tune</span> Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.4)
    confidence = st.slider("Face confidence", 0.5, 1.0, 0.7)
    st.markdown("---")
    st.markdown('<div class="sidebar-title"><span class="material-symbols-outlined">group</span> Class roster</div>', unsafe_allow_html=True)
    for s in STUDENT_ROSTER:
        st.markdown(f'<div class="sidebar-student"><span class="material-symbols-outlined">person</span>{s}</div>', unsafe_allow_html=True)

# -------------------------
# Functions
# -------------------------
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
    chosen_bg = random.choice(available_backgrounds)
    bg = cv2.imread(chosen_bg)
    if bg is None:
        st.error("Could not load background")
        st.stop()
    bg = cv2.resize(bg, (900, 600), interpolation=cv2.INTER_CUBIC)
    students = os.listdir(REFERENCE_DIR)
    num_present = random.randint(0, len(students))
    present = random.sample(students, num_present)
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
                img_path = os.path.join(student_dir, random.choice(imgs))
                face = cv2.imread(img_path)
                if face is not None:
                    new_w = int(cell_w * 0.8)
                    new_h = int(cell_h * 0.8)
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_no_bg = remove(face_pil).resize((new_w, new_h))
                    x, y = positions[i]
                    x = x + (cell_w - new_w) // 2
                    y = y + (cell_h - new_h) // 2
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
    with st.spinner("Scanning faces..."):
        faces, original_img_rgb = extract_faces(image_pil, confidence_threshold)

    st.markdown(f'<p style="color:#b09080;font-size:13px;margin-bottom:1rem;display:flex;align-items:center;gap:5px;"><span class="material-symbols-outlined" style="font-size:16px;color:#c99566;">center_focus_strong</span> {len(faces)} faces detected</p>', unsafe_allow_html=True)

    present_students = {}
    recognized_faces = []

    for i, data in enumerate(faces):
        img = data["face"]
        box = data["box"]
        try:
            img_array = np.array(img)
            result = DeepFace.represent(
                img_path=img_array,
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
            dists = [cosine_distance(emb, ref_emb) for ref_emb in ref_embs]
            avg_distances[name] = min(dists)

        best_name, best_dist = min(avg_distances.items(), key=lambda x: x[1])
        if best_dist > threshold:
            best_name = None

        if best_name and best_name not in present_students:
            present_students[best_name] = img
            recognized_faces.append({"name": best_name, "box": box, "dist": best_dist})

    # Draw boxes
    img_draw = Image.fromarray(original_img_rgb)
    draw = ImageDraw.Draw(img_draw)
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    font_name = font_conf = None
    for path in font_paths:
        if os.path.exists(path):
            font_name = ImageFont.truetype(path, 32)
            font_conf = ImageFont.truetype(path, 20)
            break
    if not font_name:
        font_name = ImageFont.load_default(size=32)
        font_conf = ImageFont.load_default(size=20)

    for face in recognized_faces:
        x, y, w, h = face["box"]
        confidence_pct = int((1 - face["dist"]) * 100)
        draw.rectangle([x, y, x+w, y+h], outline=(201,149,102), width=3)
        draw.text((x, y-42), face["name"], fill=(181,120,74), font=font_name)
        draw.text((x, y-20), f"{confidence_pct}%", fill=(212,168,83), font=font_conf)

    st.image(img_draw, use_column_width=True)

    missing_students = [s for s in STUDENT_ROSTER if s not in present_students]
    pct = int(len(present_students) / len(STUDENT_ROSTER) * 100)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">
                <span class="material-symbols-outlined" style="font-size:14px;color:#7a9e6a;">check_circle</span>
                Present
            </div>
            <div class="stat-val stat-green">{len(present_students)}</div>
            <div class="stat-sub">out of {len(STUDENT_ROSTER)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">
                <span class="material-symbols-outlined" style="font-size:14px;color:#c4605a;">cancel</span>
                Absent
            </div>
            <div class="stat-val stat-red">{len(missing_students)}</div>
            <div class="stat-sub">check required</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">
                <span class="material-symbols-outlined" style="font-size:14px;color:#d4a853;">insights</span>
                Attendance
            </div>
            <div class="stat-val stat-gold">{pct}%</div>
            <div class="stat-sub">today</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Present
    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-present"><span class="material-symbols-outlined">how_to_reg</span> Present</span><div class="divider-line"></div></div>', unsafe_allow_html=True)
    if present_students:
        cols = st.columns(5)
        for i, (name, img) in enumerate(present_students.items()):
            with cols[i % 5]:
                st.image(img, width=100)
                st.markdown(f'<div style="text-align:center;color:#7a9e6a;font-weight:600;font-size:13px;">{name}</div>', unsafe_allow_html=True)

    # Absent
    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-absent"><span class="material-symbols-outlined">person_off</span> Absent</span><div class="divider-line"></div></div>', unsafe_allow_html=True)
    if missing_students:
        cols = st.columns(5)
        for i, name in enumerate(missing_students):
            with cols[i % 5]:
                if name in reference_photos:
                    st.image(reference_photos[name], width=100)
                st.markdown(f'<div style="text-align:center;color:#c4605a;font-weight:600;font-size:13px;">{name}</div>', unsafe_allow_html=True)
    else:
        st.success("Everyone's here today!")

# -------------------------
# Mode selection
# -------------------------
mode = st.radio(
    "",
    ["upload  Upload Photo", "shuffle  Random Class", "photo_camera  Live Camera"],
    horizontal=True
)

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

if mode == "upload  Upload Photo":
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon"><span class="material-symbols-outlined">cloud_upload</span></div>
        <div class="upload-text">Drop your class photo here</div>
        <div class="upload-sub">JPG · PNG · JPEG</div>
    </div>
    """, unsafe_allow_html=True)
    class_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if st.button("Scan for Attendance"):
        if class_file is None:
            st.warning("Please upload a photo first")
            st.stop()
        class_image = Image.open(class_file)
        class_image = ImageOps.exif_transpose(class_image)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        recognize_faces(class_image, confidence, threshold)

elif mode == "shuffle  Random Class":
    st.markdown('<p class="mode-desc">Generate a random class photo with students placed on a classroom background.</p>', unsafe_allow_html=True)
    if st.button("Generate Class Photo"):
        with st.spinner("Generating class photo..."):
            result_img, present = generate_class_image()
        pil_image = Image.fromarray(result_img)
        st.image(pil_image, use_column_width=True)
        present_str = ", ".join(present) if present else "Nobody"
        st.markdown(f'<p style="color:#b09080;font-size:13px;margin:8px 0;">Actually present: <span style="color:#c99566;font-weight:600;">{present_str}</span></p>', unsafe_allow_html=True)
        st.markdown("---")
        recognize_faces(pil_image, confidence, threshold)

elif mode == "photo_camera  Live Camera":
    st.markdown('<p class="mode-desc">Take a photo directly from your camera.</p>', unsafe_allow_html=True)
    camera_photo = st.camera_input("")
    if camera_photo is not None:
        class_image = Image.open(camera_photo)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        if st.button("Scan for Attendance"):
            recognize_faces(class_image, confidence, threshold)
