import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Kantuk", layout="centered")

# --- LOAD MODEL (Dicache agar cepat) ---
@st.cache_resource
def load_keras_model():
    # Pastikan file model ada di folder yang sama atau sesuaikan path-nya
    # Jika file model Anda bernama 'model_kantuk.h5', ganti di bawah ini:
    model_path = 'models/2024-11-16_02-39-31.h5' 
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di: {model_path}. Mohon upload file model .h5 Anda.")
        return None
    return load_model(model_path)

model = load_keras_model()

# --- LOAD FACE DETECTOR BAWAAN OPENCV ---
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# --- LOGIKA PEMROSESAN VIDEO ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.score = 0
        self.rpred = [99]
        self.lpred = [99]

    def recv(self, frame):
        # Konversi frame dari WebRTC ke format OpenCV
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        # Gambar kotak di wajah (Opsional, untuk debug)
        cv2.rectangle(img, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        # --- LOGIKA DETEKSI MATA KANAN ---
        for (x, y, w, h) in right_eye:
            r_eye = img[y:y+h, x:x+w]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            
            # Prediksi
            if model is not None:
                self.rpred = np.argmax(model.predict(r_eye), axis=-1)
            break

        # --- LOGIKA DETEKSI MATA KIRI ---
        for (x, y, w, h) in left_eye:
            l_eye = img[y:y+h, x:x+w]
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            
            # Prediksi
            if model is not None:
                self.lpred = np.argmax(model.predict(l_eye), axis=-1)
            break

        # --- PERHITUNGAN SKOR & STATUS ---
        # 0 = Closed, 1 = Open (Sesuaikan dengan label model Anda)
        if self.rpred[0] == 0 and self.lpred[0] == 0:
            self.score += 1
            cv2.putText(img, "Closed", (10, height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            self.score -= 1
            cv2.putText(img, "Open", (10, height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if self.score < 0:
            self.score = 0
            
        cv2.putText(img, 'Score:' + str(self.score), (100, height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # --- TRIGGER ALARM VISUAL ---
        if self.score > 15:
            # Peringatan Visual: Kotak Merah Tebal di sekeliling layar
            cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), 20)
            # Tulisan Peringatan
            cv2.putText(img, "BANGUN!!!", (int(width/2)-100, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            # Note: Suara pygame dihapus karena tidak akan terdengar di cloud.
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- TAMPILAN STREAMLIT ---
st.title("Sistem Deteksi Kantuk Pengendara")
st.write("Pastikan izinkan akses kamera di browser Anda.")

if model is None:
    st.warning("Model belum dimuat. Pastikan file .h5 sudah diupload.")
else:
    # Konfigurasi STUN server agar bisa jalan online
    webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False}
    )