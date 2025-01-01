import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from PIL import Image
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Memuat Dataset
path = 'C:/Users/dell/latihan/tugas/text kelompok 9/Data Resep Makanan.csv'
data = pd.read_csv(path)

# 2. Pra-pemrosesan
# Menghapus kolom yang tidak diperlukan
data = data[['Title', 'Ingredients', 'Instructions', 'Image_Name']]
data = data.dropna()

# Pembersihan teks menggunakan stopwords dan lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fungsi untuk membersihkan teks
def clean_text(text):
    """
    Membersihkan teks dengan mengubah ke huruf kecil, lemmatization, dan menghapus stopwords.
    """
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(words)

# Terapkan pembersihan pada Ingredients dan Instructions
data['Ingredients'] = data['Ingredients'].apply(clean_text)
data['Instructions'] = data['Instructions'].apply(clean_text)

# 3. TF-IDF Vectorizer
# Mengubah teks menjadi representasi numerik untuk analisis kemiripan
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['Ingredients'])

# 4. KNN Model
# Melatih model KNN untuk pencarian tetangga terdekat
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(tfidf_matrix)

# 5. Pemrosesan Input Pengguna
def preprocess_input(user_input):
    """
    Memproses input dari pengguna dengan langkah yang sama seperti data pelatihan.
    """
    words = user_input.lower().split(',')
    words = [lemmatizer.lemmatize(word.strip()) for word in words if word.strip().isalpha()]
    return words

# 6. Rekomendasi KNN
def knn_recommend(user_input):
    """
    Mencari rekomendasi resep menggunakan KNN.
    """
    user_input = preprocess_input(user_input)
    user_input_str = ' '.join(user_input)
    input_vec = tfidf.transform([user_input_str])
    distances, indices = knn.kneighbors(input_vec)
    return data.iloc[indices[0]][['Title', 'Ingredients', 'Instructions', 'Image_Name']]

# 7. Antarmuka Streamlit
st.markdown(
    """
    <style>
    .left-container {
        text-align: left;
        margin-left: 20px; /* Menggeser konten ke kiri */
    }
    .left-container input {
        width: 300px; /* Lebar input box */
        margin-bottom: 10px;
    }
    .left-container button {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul dan Input di kiri
st.markdown('<div class="left-container">', unsafe_allow_html=True)
st.title("Rekomendasi Resep Makanan")
user_input = st.text_input("Masukkan bahan yang kamu punya (pisahkan dengan koma):")
if st.button("Cari Rekomendasi"):
    if user_input.strip():
        st.session_state.user_input = user_input  # Simpan input pengguna di session state
    else:
        st.warning("Masukkan bahan terlebih dahulu!")
st.markdown('</div>', unsafe_allow_html=True)

# Menampilkan hasil rekomendasi jika ada
if 'user_input' in st.session_state:
    user_input = st.session_state.user_input
    results = knn_recommend(user_input)
    
    st.header("Hasil Rekomendasi")
    for index, row in results.iterrows():
        image_folder = 'C:/Users/dell/latihan/tugas/text kelompok 9/Food Images'
        possible_extensions = ['.jpg', '.png', '.jpeg']
        img_path = None
        for ext in possible_extensions:
            temp_path = os.path.join(image_folder, row['Image_Name'] + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        # Menampilkan gambar dan detail resep
        cols = st.columns([1, 3])
        with cols[0]:
            if img_path:
                img = Image.open(img_path)
                st.image(img, width=100)
            else:
                st.text("Gambar tidak ditemukan")
        with cols[1]:
            st.write(f"**{row['Title']}**")
            st.write(f"**Ingredients:** {row['Ingredients']}")
            st.write(f"**Instructions:** {row['Instructions']}")
