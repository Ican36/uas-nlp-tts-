import streamlit as st
import os
import wave
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression


# Data latihan mini untuk klasifikasi sentimen
sentences = [
    ["saya", "suka", "aplikasi", "ini"],
    ["aplikasi", "ini", "sangat", "bagus"],
    ["mantap", "bagus", "sekali"],
    ["saya", "benci", "aplikasi", "ini"],
    ["aplikasi", "ini", "sangat", "buruk"],
    ["jelek", "sekali", "aplikasi", "ini"]
]
# 1 = Positif, 0 = Negatif
labels = [1, 1, 1, 0, 0, 0] 

# Latih model Word2Vec (Word Embedding) secara instan
w2v_model = Word2Vec(sentences, vector_size=20, min_count=1, window=2, epochs=50)

# Fungsi mengubah kalimat menjadi vektor rata-rata
def get_sentence_vector(sentence, model):
    words = sentence.lower().split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(20)
    return np.mean(vectors, axis=0)

# Ubah semua data latihan menjadi vektor
X_train = np.array([get_sentence_vector(" ".join(s), w2v_model) for s in sentences])
y_train = np.array(labels)

# Latih Classifier sederhana (Logistic Regression)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# ==========================================
# 2. LOGIKA TTS (TEKS TO SPEECH) CUSTOM
# ==========================================

from pydub import AudioSegment

def gabung_audio(list_kata, output_filename="output.wav"):
    """Menggabungkan audio menggunakan wave bawaan, melewati file yang error"""
    import wave
    data = []
    folder_suara = "suara"
    
    for kata in list_kata:
        # Coba cari file .wav di folder
        file_path = os.path.join(folder_suara, f"{kata.lower()}.wav")
        if os.path.exists(file_path):
            try:
                w = wave.open(file_path, 'rb')
                data.append([w.getparams(), w.readframes(w.getnframes())])
                w.close()
            except Exception:
                # Jika file corrupt / bukan WAV asli, dilewati agar tidak error crash
                continue
            
    if not data:
        return None

    # Tulis hasil gabungan
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
st.subheader("Fitur: Word2Vec Embedding + Audio Stitching")
st.write("---")

# Input Teks dari Pengguna
user_text = st.text_input("Ketik kalimat di sini (Contoh: saya suka aplikasi bagus):")

if st.button("Proses Kalimat"):
    if user_text:
       
        vektor_input = get_sentence_vector(user_text, w2v_model).reshape(1, -1)
        prediksi = classifier.predict(vektor_input)[0]
        hasil_sentimen = "POSITIF 😊" if prediksi == 1 else "NEGATIF 😡"
        
        # Tampilkan Hasil Klasifikasi
        st.metric(label="Hasil Analisis Sentimen:", value=hasil_sentimen)
        st.write("*(Klasifikasi ini diproses menggunakan vektor dari Word2Vec)*")
        
        # ---- PROSES TTS CUSTOM ----
        st.write("### 🔊 Mendengarkan Suara Mahasiswa:")
        kata_kata = user_text.split()
        
        # Jalankan fungsi penggabung audio
        file_audio_hasil = gabung_audio(kata_kata)
        
        if file_audio_hasil:
            # Putar audio langsung di web browser
            st.audio(file_audio_hasil, format="audio/wav")
            st.success("Suara berhasil digenerate berdasarkan rekaman kamu!")
        else:
            st.warning("Maaf, kata yang kamu ketik belum ada rekaman suaranya di folder 'suara/'.")
    else:
        st.error("Silakan ketik sesuatu terlebih dahulu!")