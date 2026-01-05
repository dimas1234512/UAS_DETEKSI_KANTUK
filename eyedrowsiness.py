import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import tempfile # Wajib untuk memproses upload video
import os

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Deteksi Kantuk Multi-Input", layout="wide")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file model 'deteksi-kantuk.tflite' ada di folder yang sama
    interpreter = tf.lite.Interpreter(model_path="deteksi-kantuk.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error("Model 'deteksi-kantuk.tflite' tidak ditemukan! Pastikan file ada satu folder dengan app.py")
    st.stop()

# --- 2. SETUP WAJAH & MATA ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- 3. FUNGSI SUARA (BEEP) ---
def play_alarm():
    sound_html = """
    <audio autoplay>
    <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mp3">
    </audio>
    """
    st.components.v1.html(sound_html, height=0, width=0)

# --- 4. PREDIKSI ---
def predict_eye(eye_img):
    img_resized = cv2.resize(eye_img, (64, 64))
    input_data = np.array(img_resized, dtype=np.float32)
    input_data = input_data / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# --- 5. FUNGSI INTI PEMROSESAN FRAME (REUSABLE) ---
def process_frame(frame):
    """
    Fungsi ini menerima frame gambar (BGR), mendeteksi wajah/mata,
    menggambar kotak, dan mengembalikan status.
    Digunakan untuk Kamera, Gambar, dan Video.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    status_frame = "TIDAK_TAHU"
    box_color = (100, 100, 100)
    avg_score = 0
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = rgb_frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        probs = []
        
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            try:
                pred = predict_eye(eye_img)
                probs.append(pred)
                
                # Kotak Mata (Hijau=Buka, Merah=Tutup)
                e_color = (0, 255, 0) if pred > 0.5 else (255, 0, 0)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), e_color, 2)
            except:
                pass
        
        if len(probs) > 0:
            avg_score = sum(probs) / len(probs)
            
            if avg_score < 0.5:
                status_frame = "TERTUTUP"
                box_color = (255, 255, 0) # Kuning
            else:
                status_frame = "TERBUKA"
                box_color = (0, 255, 0) # Hijau

        # Gambar Kotak Wajah
        cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), box_color, 3)
        
    return rgb_frame, status_frame, avg_score

# ==========================================
# TAMPILAN UTAMA (UI)
# ==========================================

st.sidebar.title("🔧 Pengaturan")

# PILIHAN SUMBER INPUT
input_source = st.sidebar.radio("Pilih Sumber Input:", ("Kamera Realtime", "Upload Gambar", "Upload Video"))

alarm_threshold = st.sidebar.slider("Waktu Tunggu (Detik) sebelum Alarm:", 1.0, 10.0, 3.0, 0.5) 

st.title("👁️ Deteksi Kantuk Multi-Input")
st.write(f"Mode Aktif: **{input_source}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Status Saat Ini**")
    status_text = st.empty()
with col2:
    st.markdown("**Skor Mata Terbuka**")
    kpi_text = st.empty()
with col3:
    st.markdown("**⏳ Timer Mata Tertutup**")
    timer_text = st.empty()

frame_window = st.image([])

# ==========================================
# LOGIKA LOOPING (VIDEO & KAMERA)
# ==========================================

# Variabel Timer Global untuk loop
start_time_closed = None 

def run_detection_loop(cap, is_video_file=False):
    """Fungsi Loop utama untuk memproses video frame-by-frame"""
    global start_time_closed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                st.info("Video selesai.") 
            else:
                st.error("Gagal membaca kamera.")
            break
            
        # Mirror hanya untuk kamera webcam, jangan untuk video file
        if not is_video_file:
            frame = cv2.flip(frame, 1)

        # --- PANGGIL FUNGSI PROSES ---
        processed_frame, status, score = process_frame(frame)
        
        # --- LOGIKA TIMER ---
        duration_closed = 0
        final_status_text = ""

        if status == "TERTUTUP":
            if start_time_closed is None:
                start_time_closed = time.time()
            duration_closed = time.time() - start_time_closed
            final_status_text = "Mata Tertutup..."
            
        elif status == "TERBUKA":
            start_time_closed = None # RESET
            duration_closed = 0
            final_status_text = "✅ AMAN"
            
        else: # TIDAK_TAHU (Wajah hilang)
            if start_time_closed is not None:
                duration_closed = time.time() - start_time_closed
                final_status_text = "⚠️ Wajah Hilang (Timer Lanjut...)"
            else:
                final_status_text = "Mencari Wajah..."

        # --- ALARM & VISUALISASI ---
        if duration_closed > alarm_threshold:
            final_status_text = "⚠️ BAHAYA: NGANTUK!"
            # Teks Merah Besar
            cv2.putText(processed_frame, f"BANGUN! ({duration_closed:.1f}s)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
            # Bingkai Merah
            cv2.rectangle(processed_frame, (0,0), (processed_frame.shape[1], processed_frame.shape[0]), (255,0,0), 20)
            play_alarm()
            
        elif start_time_closed is not None:
             # Timer Kuning
             cv2.putText(processed_frame, f"Timer: {duration_closed:.1f}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- UPDATE UI ---
        frame_window.image(processed_frame)
        
        if "BAHAYA" in final_status_text: status_text.error(final_status_text)
        elif "Tertutup" in final_status_text: status_text.warning(final_status_text)
        elif "AMAN" in final_status_text: status_text.success(final_status_text)
        else: status_text.info(final_status_text)
        
        kpi_text.metric("Skor", f"{int(score*100)} %")
        timer_text.metric("Timer", f"{duration_closed:.2f} s")

        # Jika video file, tambahkan sedikit delay agar tidak terlalu cepat
        # (Opsional, tergantung performa)
        if is_video_file:
            time.sleep(0.01) 

# ==========================================
# EKSEKUSI BERDASARKAN PILIHAN INPUT
# ==========================================

# --- MODE 1: KAMERA REALTIME ---
if input_source == "Kamera Realtime":
    run_btn = st.sidebar.checkbox("Mulai Kamera", value=False)
    if run_btn:
        cap = cv2.VideoCapture(0)
        run_detection_loop(cap, is_video_file=False)
        cap.release()
    else:
        st.info("Centang 'Mulai Kamera' di sidebar untuk memulai.")

# --- MODE 2: UPLOAD GAMBAR ---
elif input_source == "Upload Gambar":
    uploaded_file = st.sidebar.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert file upload ke OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Proses 1 Frame saja
        processed_frame, status, score = process_frame(frame)
        
        # Tampilkan
        frame_window.image(processed_frame)
        st.write(f"**Hasil Deteksi:** {status} (Skor Mata Terbuka: {int(score*100)}%)")
        
        if status == "TERTUTUP":
            st.error("⚠️ Mata Terdeteksi Tertutup!")
        elif status == "TERBUKA":
            st.success("✅ Mata Terbuka.")
        else:
            st.warning("❓ Wajah tidak terdeteksi dengan jelas.")

# --- MODE 3: UPLOAD VIDEO ---
elif input_source == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Simpan video ke temporary file agar bisa dibaca OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        st.write("Sedang memutar video...")
        run_detection_loop(cap, is_video_file=True)
        
        cap.release()