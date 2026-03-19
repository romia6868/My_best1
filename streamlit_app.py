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
    page_title="מערכת נוכחות חכמה",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CSS מותאם אישית
# -------------------------
st.markdown("""
<style>
    /* רקע כללי */
    .stApp { background: #0f1117; }
    
    /* כותרת ראשית */
    .main-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid #ffffff15;
        margin-bottom: 1.5rem;
    }
    .header-icon {
        width: 46px; height: 46px;
        background: #7C3AED;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
    }
    .header-title { font-size: 26px; font-weight: 700; color: white; margin: 0; }
    .header-sub { font-size: 13px; color: #888; margin: 2px 0 0; }

    /* כרטיסי סטטיסטיקה */
    .stat-row { display: flex; gap: 12px; margin: 1.5rem 0; }
    .stat-card {
        flex: 1;
        background: #1a1d27;
        border: 1px solid #ffffff10;
        border-radius: 12px;
        padding: 16px 18px;
    }
    .stat-label { font-size: 12px; color: #888; margin-bottom: 6px; }
    .stat-val { font-size: 28px; font-weight: 700; }
    .stat-sub { font-size: 11px; color: #555; margin-top: 3px; }
    .stat-green { color: #10B981; }
    .stat-red { color: #F43F5E; }
    .stat-white { color: #fff; }

    /* טאבים */
    .stRadio > div { flex-direction: row !important; gap: 8px; }
    .stRadio label {
        background: #1a1d27 !important;
        border: 1px solid #ffffff15 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        color: #aaa !important;
        font-size: 14px !important;
    }
    .stRadio label:has(input:checked) {
        background: #7C3AED !important;
        border-color: #7C3AED !important;
        color: white !important;
    }

    /* כפתור ראשי */
    .stButton > button {
        background: #7C3AED !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #6D28D9 !important;
        transform: translateY(-1px) !important;
    }

    /* אזור העלאה */
    .upload-zone {
        border: 2px dashed #7C3AED55;
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        background: #7C3AED08;
        margin-bottom: 1rem;
    }
    .upload-icon { font-size: 36px; margin-bottom: 10px; }
    .upload-text { font-size: 15px; color: #aaa; margin-bottom: 4px; }
    .upload-sub { font-size: 12px; color: #555; }

    /* כרטיסי תלמידים */
    .student-card {
        background: #1a1d27;
        border: 1px solid #ffffff10;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        transition: transform 0.2s;
    }
    .student-card:hover { transform: translateY(-2px); }
    .student-avatar {
        width: 56px; height: 56px;
        border-radius: 50%;
        margin: 0 auto 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; font-weight: 700;
    }
    .avatar-present { background: #10B98120; color: #10B981; }
    .avatar-absent { background: #F43F5E20; color: #F43F5E; }
    .student-name { font-size: 13px; font-weight: 600; color: white; }
    .student-pct { font-size: 11px; color: #38BDF8; margin-top: 3px; }
    .student-absent-label { font-size: 11px; color: #F43F5E66; margin-top: 3px; }

    /* מפריד */
    .section-divider {
        display: flex; align-items: center; gap: 12px;
        margin: 1.8rem 0 1.2rem;
    }
    .divider-line { flex: 1; height: 1px; background: #ffffff10; }
    .divider-badge {
        font-size: 12px; padding: 3px 12px;
        border-radius: 20px; font-weight: 600;
    }
    .badge-present { background: #10B98120; color: #10B981; }
    .badge-absent { background: #F43F5E20; color: #F43F5E; }

    /* סיידבר */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #13151f !important;
        border-right: 1px solid #ffffff10 !important;
    }
    .sidebar-title { font-size: 16px; font-weight: 700; color: white; margin-bottom: 1rem; }
    .sidebar-student {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 10px;
        background: #1a1d27;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 13px; color: #ddd;
    }
    .student-dot { width: 8px; height: 8px; border-radius: 50%; background: #7C3AED; }

    /* spinner */
    .stSpinner { color: #7C3AED !important; }

    /* תמונת זיהוי */
    .result-image { border-radius: 14px; overflow: hidden; border: 1px solid #ffffff15; }

    /* info/success/warning */
    .stAlert { border-radius: 10px !important; }
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
                    except Exception as e:
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
# כותרת
# -------------------------
st.markdown("""
<div class="main-header">
    <div class="header-icon">📸</div>
    <div>
        <div class="header-title">מערכת נוכחות חכמה</div>
        <div class="header-sub">זיהוי פנים אוטומטי · Facenet512 · RetinaFace</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# סיידבר
# -------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ הגדרות</div>', unsafe_allow_html=True)
    threshold = st.slider("רמת זיהוי", 0.0, 1.0, 0.4, help="ככל שנמוך יותר – קפדני יותר")
    confidence = st.slider("רגישות זיהוי פנים", 0.5, 1.0, 0.7)
    st.markdown("---")
    st.markdown('<div class="sidebar-title">👥 רשימת כיתה</div>', unsafe_allow_html=True)
    for s in STUDENT_ROSTER:
        st.markdown(f'<div class="sidebar-student"><div class="student-dot"></div>{s}</div>', unsafe_allow_html=True)

# -------------------------
# פונקציות
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
        st.error("לא נמצאו תמונות רקע")
        st.stop()
    chosen_bg = random.choice(available_backgrounds)
    bg = cv2.imread(chosen_bg)
    if bg is None:
        st.error(f"לא ניתן לטעון את הרקע")
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
        st.warning(f"שגיאה בזיהוי פנים: {e}")
    return faces, img_rgb

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def recognize_faces(image_pil, confidence_threshold=0.7, threshold=0.4):
    with st.spinner("מנתח פנים..."):
        faces, original_img_rgb = extract_faces(image_pil, confidence_threshold)

    st.markdown(f'<p style="color:#888; font-size:13px; margin-bottom:1rem;">זוהו {len(faces)} פנים בתמונה</p>', unsafe_allow_html=True)

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

    # ציור על תמונה
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
        draw.rectangle([x, y, x+w, y+h], outline=(124,58,237), width=3)
        draw.text((x, y-42), face["name"], fill=(124,58,237), font=font_name)
        draw.text((x, y-20), f"{confidence_pct}%", fill=(56,189,248), font=font_conf)

    st.image(img_draw, use_column_width=True)

    # סטטיסטיקות
    missing_students = [s for s in STUDENT_ROSTER if s not in present_students]
    pct = int(len(present_students) / len(STUDENT_ROSTER) * 100)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">נוכחים</div>
            <div class="stat-val stat-green">{len(present_students)}</div>
            <div class="stat-sub">מתוך {len(STUDENT_ROSTER)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">חסרים</div>
            <div class="stat-val stat-red">{len(missing_students)}</div>
            <div class="stat-sub">יש לבדוק</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">אחוז נוכחות</div>
            <div class="stat-val stat-white">{pct}%</div>
            <div class="stat-sub">היום</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # נוכחים
    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-present">✅ נוכחים</span><div class="divider-line"></div></div>', unsafe_allow_html=True)

    if present_students:
        cols = st.columns(5)
        for i, (name, img) in enumerate(present_students.items()):
            with cols[i % 5]:
                st.image(img, width=100)
                st.markdown(f'<div class="student-name" style="text-align:center;color:#10B981;font-weight:600;font-size:13px;">{name}</div>', unsafe_allow_html=True)

    # חסרים
    st.markdown('<div class="section-divider"><div class="divider-line"></div><span class="divider-badge badge-absent">❌ חסרים</span><div class="divider-line"></div></div>', unsafe_allow_html=True)

    if missing_students:
        cols = st.columns(5)
        for i, name in enumerate(missing_students):
            with cols[i % 5]:
                if name in reference_photos:
                    st.image(reference_photos[name], width=100)
                st.markdown(f'<div style="text-align:center;color:#F43F5E;font-weight:600;font-size:13px;">{name}</div>', unsafe_allow_html=True)
    else:
        st.success("כולם נוכחים היום! 🎉")

# -------------------------
# בחירת מצב
# -------------------------
mode = st.radio("", ["📤 העלאת תמונה", "🎲 הגרלה רנדומלית", "📷 צילום מצלמה"], horizontal=True)

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

if mode == "📤 העלאת תמונה":
    st.markdown('<div class="upload-zone"><div class="upload-icon">🖼️</div><div class="upload-text">גרור תמונת כיתה לכאן</div><div class="upload-sub">JPG · PNG · JPEG</div></div>', unsafe_allow_html=True)
    class_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if st.button("🔍 בדוק נוכחות"):
        if class_file is None:
            st.warning("יש להעלות תמונה תחילה")
            st.stop()
        class_image = Image.open(class_file)
        class_image = ImageOps.exif_transpose(class_image)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        recognize_faces(class_image, confidence, threshold)

elif mode == "🎲 הגרלה רנדומלית":
    st.markdown('<p style="color:#888; font-size:14px;">יצירת תמונת כיתה רנדומלית עם תלמידים אקראיים</p>', unsafe_allow_html=True)
    if st.button("🎲 צור תמונת כיתה"):
        with st.spinner("מייצר תמונת כיתה..."):
            result_img, present = generate_class_image()
        pil_image = Image.fromarray(result_img)
        st.image(pil_image, use_column_width=True)
        st.markdown(f'<p style="color:#888; font-size:13px; margin: 8px 0;">נוכחים אמיתיים: <span style="color:#7C3AED; font-weight:600;">{", ".join(present) if present else "אף אחד"}</span></p>', unsafe_allow_html=True)
        st.markdown("---")
        recognize_faces(pil_image, confidence, threshold)

elif mode == "📷 צילום מצלמה":
    st.markdown('<p style="color:#888; font-size:14px;">צלמי את תמונת הכיתה ישירות מהמצלמה</p>', unsafe_allow_html=True)
    camera_photo = st.camera_input("")
    if camera_photo is not None:
        class_image = Image.open(camera_photo)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        if st.button("🔍 בדוק נוכחות"):
            recognize_faces(class_image, confidence, threshold)
