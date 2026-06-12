import streamlit as st
import os
import wave
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. PREPARASI DATA & TEXT EMBEDDING (NLP)
# ==========================================

# Data latihan mini untuk klasifikasi sentimen
sentences = [
    "saya suka aplikasi ini",
    "aplikasi ini sangat bagus",
    "mantap bagus sekali",
    "saya benci aplikasi ini",
    "aplikasi ini sangat buruk",
    "jelek sekali aplikasi ini"
]
# 1 = Positif, 0 = Negatif
labels = [1, 1, 1, 0, 0, 0] 

# Menggunakan TF-IDF Vectorizer sebagai pengganti Word2Vec (Lebih ringan & aman untuk Cloud)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(sentences).toarray()
y_train = np.array(labels)

# Latih Classifier sederhana (Logistic Regression)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# ==========================================
# 2. LOGIKA TTS (TEKS TO SPEECH) DARURAT
# ==========================================

def gabung_audio(list_kata, output_filename="output.wav"):
    """Menggabungkan audio menggunakan wave bawaan (Tanpa FFmpeg agar aman di cloud)"""
    data = []
    folder_suara = "suara"
    
    for kata in list_kata:
        # Menghapus tanda baca jika ada
        kata_bersih = "".join(c for c in kata if c.isalnum())
        file_path = os.path.join(folder_suara, f"{kata_bersih.lower()}.wav")
        
        if os.path.exists(file_path):
            try:
                w = wave.open(file_path, 'rb')
                data.append([w.getparams(), w.readframes(w.getnframes())])
                w.close()
            except Exception:
                continue
            
    if not data:
        return None

    output = wave.open(output_filename, 'wb')
    output.setparams(data[0][0])
    for item in data:
        output.writeframes(item[1])
    output.close()
    return output_filename


# ==========================================
# 3. TAMPILAN APLIKASI (STREAMLIT UI)
# ==========================================

st.set_page_config(page_title="UAS NLP & TTS", layout="centered")
st.title("🎓 Aplikasi Klasifikasi Sentimen & TTS Custom")
st.subheader("Fitur: TF-IDF Text Embedding + Audio Stitching")
st.write("---")

user_text = st.text_input("Ketik kalimat di sini (Contoh: saya suka aplikasi bagus):")

if st.button("Proses Kalimat"):
    if user_text:
        # ---- PROSES NLP & KLASIFIKASI ----
        vektor_input = vectorizer.transform([user_text]).toarray()
        prediksi = classifier.predict(vektor_input)[0]
        hasil_sentimen = "POSITIF 😊" if prediksi == 1 else "NEGATIF 😡"
        
        st.metric(label="Hasil Analisis Sentimen:", value=hasil_sentimen)
        st.write("*(Klasifikasi ini diproses menggunakan matriks embedding dari TF-IDF Vectorizer)*")
        
        # ---- PROSES TTS CUSTOM ----
        st.write("### 🔊 Mendengarkan Suara Mahasiswa:")
        kata_kata = user_text.split()
        
        file_audio_hasil = gabung_audio(kata_kata)
        
        if file_audio_hasil:
            st.audio(file_audio_hasil, format="audio/wav")
            st.success("Suara berhasil digenerate berdasarkan rekaman kamu!")
        else:
            st.warning("Maaf, kata yang kamu ketik belum ada rekaman suaranya dalam format .wav asli di folder 'suara/'.")
    else:
        st.error("Silakan ketik sesuatu terlebih dahulu!")
