import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import os
import zipfile
import random
import cv2

st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")
st.title("📸 מערכת נוכחות חכמה")

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
                        st.warning(f"שגיאה ב-{file}: {e}")
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
st.info(f"נמצאו {len(reference_embeddings)} תלמידים במאגר")

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
        st.error(f"לא ניתן לטעון את הרקע: {chosen_bg}")
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
                    face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    x, y = positions[i]
                    x = x + (cell_w - new_w) // 2
                    y = y + (cell_h - new_h) // 2
                    bg[y:y+new_h, x:x+new_w] = face
                    i += 1
    return bg, present

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
    faces, original_img_rgb = extract_faces(image_pil, confidence_threshold)
    st.write(f"זוהו {len(faces)} פנים")

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
        except Exception as e:
            st.warning(f"שגיאה בפנים {i+1}: {e}")
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

    # ציור תיבות – הכל בתוך הפונקציה
    img_draw = Image.fromarray(original_img_rgb)
    draw = ImageDraw.Draw(img_draw)

    try:
        font_name = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_conf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_name = ImageFont.load_default()
        font_conf = ImageFont.load_default()

    for face in recognized_faces:
        x, y, w, h = face["box"]
        confidence_pct = int((1 - face["dist"]) * 100)
        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=3)
        draw.text((x, y-50), face["name"], fill=(0,255,0), font=font_name)
        draw.text((x, y-22), f"{confidence_pct}%", fill=(255,255,0), font=font_conf)

    st.subheader("תוצאת זיהוי")
    st.image(img_draw, use_column_width=True)

    missing_students = [s for s in STUDENT_ROSTER if s not in present_students]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.header(f"✅ נוכחים ({len(present_students)})")
        cols = st.columns(3)
        for i, (name, img) in enumerate(present_students.items()):
            with cols[i % 3]:
                st.write(f"**{name}**")
                st.image(img, width=90)

    with col2:
        st.header(f"❌ חסרים ({len(missing_students)})")
        if missing_students:
            cols = st.columns(3)
            for i, name in enumerate(missing_students):
                with cols[i % 3]:
                    st.write(f"**{name}**")
                    if name in reference_photos:
                        st.image(reference_photos[name], width=90)
        else:
            st.success("כולם נוכחים")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("הגדרות")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.4)
    confidence = st.slider("Face Detection Confidence", 0.5, 1.0, 0.7)
    st.write("תלמידים בכיתה")
    for s in STUDENT_ROSTER:
        st.write(s)

# -------------------------
# בחירת מצב
# -------------------------
st.subheader("בחרי מצב")
mode = st.radio("", ["📤 העלאת תמונה", "🎲 הגרלת תמונה רנדומלית"], horizontal=True)

if mode == "📤 העלאת תמונה":
    class_file = st.file_uploader("Upload class photo", type=["jpg","jpeg","png"])
    if st.button("בדוק נוכחות"):
        if class_file is None:
            st.warning("יש להעלות תמונה")
            st.stop()
        class_image = Image.open(class_file)
        class_image = ImageOps.exif_transpose(class_image)
        if max(class_image.size) > 1200:
            class_image.thumbnail((1200, 1200))
        recognize_faces(class_image, confidence, threshold)

elif mode == "🎲 הגרלת תמונה רנדומלית":
    if st.button("צור תמונת כיתה רנדומלית"):
        bg_img, present = generate_class_image()
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(bg_rgb)
        st.subheader("התמונה הרנדומלית שנוצרה")
        st.image(pil_image, use_column_width=True)
        st.write("נוכחים אמיתיים:", present)
        st.divider()
        st.subheader("תוצאות הזיהוי האוטומטי")
        recognize_faces(pil_image, confidence, threshold)
