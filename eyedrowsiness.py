import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
from PIL import Image

# --- SETUP HALAMAN SEDERHANA ---
st.set_page_config(page_title="Deteksi Kantuk + Upload Gambar", layout="wide")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file model ada di folder yang sama
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

# --- 5. FUNGSI PEMROSESAN FRAME (BARU - REUSABLE) ---
# Fungsi ini memisahkan logika deteksi agar bisa dipakai Kamera & Upload Gambar
def process_frame(frame):
    # Konversi Warna
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    status_frame = "TIDAK_TAHU" # Default
    box_color = (100, 100, 100)
    avg_score = 0
    
    # Deteksi Wajah & Mata
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
                
                # Kotak Mata
                e_color = (0, 255, 0) if pred > 0.5 else (255, 0, 0)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), e_color, 2)
            except:
                pass
            
        if len(probs) > 0:
            avg_score = sum(probs) / len(probs)
            
            # Tentukan status frame ini
            if avg_score < 0.5: 
                status_frame = "TERTUTUP"
                box_color = (255, 255, 0) 
            else:
                status_frame = "TERBUKA"
                box_color = (0, 255, 0)

        # Gambar Kotak Wajah
        cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), box_color, 3)
        
    return rgb_frame, status_frame, avg_score

# ==========================================
# TAMPILAN UTAMA (UI)
# ==========================================

st.sidebar.title("🔧 Pengaturan")

# PILIHAN INPUT (BARU)
input_source = st.sidebar.radio("Pilih Sumber Input:", ("Kamera Realtime", "Upload Gambar"))

st.sidebar.write("Atur sensitivitas waktu (Khusus Kamera):")
# Slider Waktu Tunggu (Default 3 Detik)
alarm_threshold = st.sidebar.slider("Waktu Tunggu (Detik) sebelum Alarm:", 1.0, 10.0, 3.0, 0.5) 

st.title("👁️ Deteksi Kantuk Pengemudi")

if input_source == "Kamera Realtime":
    st.info(f"Sistem stabil: Jika wajah goyang, timer tetap lanjut. Alarm bunyi setelah **{alarm_threshold} detik**.")
else:
    st.info("Mode Upload Gambar: Mendeteksi status mata pada foto statis.")

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
# LOGIKA UTAMA (BERDASARKAN PILIHAN INPUT)
# ==========================================

# --- OPSI 1: KAMERA REALTIME (Fitur Asli) ---
if input_source == "Kamera Realtime":
    run_camera = st.sidebar.checkbox("Mulai Kamera", value=False)
    
    if run_camera:
        camera = cv2.VideoCapture(0)
        start_time_closed = None 
        
        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Gagal membaca kamera.")
                break
            
            # Mirror frame untuk webcam agar natural (Hanya di mode kamera)
            frame = cv2.flip(frame, 1)

            # --- PANGGIL FUNGSI PROSES FRAME ---
            processed_frame, status_frame_ini, score_display = process_frame(frame)
            score_percent = int(score_display * 100)

            # --- LOGIKA TIMER PINTAR (ANTI-RESET) ---
            # (Tidak ada yang diubah dari logika asli, hanya variabel diambil dari fungsi process_frame)
            duration_closed = 0
            final_status_text = ""

            # KONDISI 1: Mata Jelas TERTUTUP
            if status_frame_ini == "TERTUTUP":
                if start_time_closed is None:
                    start_time_closed = time.time() # Mulai timer
                
                duration_closed = time.time() - start_time_closed
                final_status_text = "Mata Tertutup..."

            # KONDISI 2: Mata Jelas TERBUKA
            elif status_frame_ini == "TERBUKA":
                start_time_closed = None # RESET Timer
                duration_closed = 0
                final_status_text = "✅ AMAN"

            # KONDISI 3: Wajah Hilang / Goyang (TIDAK_TAHU)
            else:
                if start_time_closed is not None:
                    duration_closed = time.time() - start_time_closed
                    final_status_text = "⚠️ Wajah Hilang (Timer Lanjut...)"
                else:
                    final_status_text = "Mencari Wajah..."

            # --- CEK ALARM ---
            if duration_closed > alarm_threshold:
                final_status_text = "⚠️ BAHAYA: NGANTUK!"
                
                # Visualisasi Merah (Alarm)
                cv2.putText(processed_frame, f"BANGUN! ({duration_closed:.1f}s)", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
                cv2.rectangle(processed_frame, (0,0), (640,480), (255,0,0), 10) 
                
                play_alarm()

            elif start_time_closed is not None:
                # Visualisasi Kuning (Timer Berjalan)
                cv2.putText(processed_frame, f"Timer: {duration_closed:.1f}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Tampilkan Video
            frame_window.image(processed_frame)
            
            # Update Teks UI
            if "BAHAYA" in final_status_text:
                status_text.error(final_status_text)
            elif "Tertutup" in final_status_text:
                status_text.warning(final_status_text)
            elif "Lanjut" in final_status_text:
                status_text.warning(final_status_text)
            elif "AMAN" in final_status_text:
                status_text.success(final_status_text)
            else:
                status_text.info(final_status_text)
                
            kpi_text.metric("Skor Mata", f"{score_percent} %")
            
            if duration_closed > 0:
                 timer_text.metric("Timer", f"{duration_closed:.2f} s")
            else:
                 timer_text.metric("Timer", "0.00 s")

        camera.release()

# --- OPSI 2: UPLOAD GAMBAR (Fitur Tambahan) ---
elif input_source == "Upload Gambar":
    uploaded_file = st.sidebar.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Konversi file upload ke format OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Panggil fungsi proses frame (Sama dengan kamera, tapi 1x jalan)
        # Note: Tidak di-flip agar teks/gambar tetap terbaca benar
        processed_frame, status, score = process_frame(frame)
        
        # Tampilkan Hasil
        frame_window.image(processed_frame)
        
        # Update Metrics
        kpi_text.metric("Skor Mata", f"{int(score * 100)} %")
        timer_text.metric("Timer", "N/A (Mode Gambar)")
        
        if status == "TERTUTUP":
            status_text.error("Terdeteksi: MATA TERTUTUP")
            st.error("⚠️ Peringatan: Mata terindikasi mengantuk pada foto ini.")
        elif status == "TERBUKA":
            status_text.success("Terdeteksi: AMAN")
            st.success("✅ Mata terdeteksi terbuka.")
        else:
            status_text.info("Wajah tidak terdeteksi")
            st.warning("Tidak dapat menemukan wajah atau mata yang jelas pada gambar.")