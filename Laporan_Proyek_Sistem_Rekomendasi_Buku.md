# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai platform digital, termasuk e-commerce, layanan streaming, dan perpustakaan digital. Seiring dengan pertumbuhan eksponensial data dan banyaknya pilihan yang tersedia, pengguna menghadapi tantangan dalam menemukan item yang sesuai dengan minat mereka secara efisien. Dalam konteks literasi digital, sistem rekomendasi buku memiliki peran strategis untuk membantu pembaca menemukan buku yang relevan dengan preferensi mereka tanpa harus mencarinya secara manual dari ribuan judul yang tersedia.

Pada proyek ini, dikembangkan sebuah sistem rekomendasi buku berbasis *Collaborative Filtering* dengan pendekatan deep learning. Metode ini memanfaatkan interaksi historis antara pengguna dan buku untuk memprediksi item yang kemungkinan besar disukai oleh pengguna, tanpa memerlukan informasi konten eksplisit dari buku tersebut.

Berdasarkan studi Bobadilla et al. (2013), *collaborative filtering* telah terbukti menjadi salah satu pendekatan paling efektif dalam sistem rekomendasi karena kemampuannya memberikan personalisasi tinggi dan relevansi yang baik. Sementara itu, pendekatan deep learning, khususnya model embedding, telah menunjukkan efektivitas dalam menangani masalah *data sparsity* dan skala besar yang umum dalam sistem rekomendasi (Zhang et al., 2019).

Dataset yang digunakan dalam proyek ini adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) dari Kaggle, yang mencakup informasi pengguna, ISBN buku, serta rating yang diberikan.

### Referensi

- Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013). Recommender systems survey. *Knowledge-Based Systems*, 46, 109–132. https://doi.org/10.1016/j.knosys.2013.03.012  
- Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. *ACM Computing Surveys (CSUR)*, 52(1), 1–38. https://doi.org/10.1145/3285029  
- Smith, A. (2017). *Amazon Recommendation Engine*. Forbes Tech. https://www.forbes.com/sites/forbestechcouncil/2017/10/16/how-amazons-recommendation-engine-works/

---

## Business Understanding

### Problem Statements
- Bagaimana membangun sistem rekomendasi buku yang dipersonalisasi menggunakan teknik collaborative filtering berbasis deep learning?
- Bagaimana sistem dapat memprediksi dan merekomendasikan buku yang kemungkinan besar disukai pengguna berdasarkan data rating sebelumnya?
- Bagaimana sistem ini dapat memberikan nilai tambah bagi pengguna dalam menemukan buku-buku yang belum pernah mereka baca sebelumnya?

### Goals
- Menghasilkan rekomendasi buku yang dipersonalisasi untuk setiap pengguna berdasarkan pola interaksi dan rating historis.
- Membangun model rekomendasi menggunakan pendekatan embedding neural network untuk collaborative filtering.
- Menyediakan prediksi rating yang akurat sebagai dasar pengambilan keputusan dalam sistem rekomendasi.

### Solution Approach

1. **Collaborative Filtering (yang digunakan)**:  
   Mengandalkan pola interaksi historis pengguna dengan buku (user–item matrix) menggunakan embedding-based neural network.

2. **Content-Based Filtering (alternatif)**:  
   Menggunakan metadata buku seperti judul, penulis, atau kategori untuk mempelajari kesamaan antar buku dan merekomendasikan buku serupa dengan yang disukai pengguna.

---

## Data Understanding

Dataset terdiri dari tiga file utama:

### 1. Books.csv
- **Dimensi awal**: 271,360 baris × 8 kolom
- **Fitur**:
  - `ISBN`: Nomor unik buku.
  - `Book-Title`: Judul buku.
  - `Book-Author`: Nama penulis.
  - `Year-Of-Publication`: Tahun terbit.
  - `Publisher`: Nama penerbit.
  - `Image-URL-S`, `Image-URL-M`, `Image-URL-L`: URL gambar buku ukuran kecil, sedang, besar.
- **Missing Value**:
  - Kolom `Book-Author`, `Publisher`, dan `Image-URL-L` memiliki nilai kosong.

### 2. Users.csv
- **Dimensi awal**: 278,858 baris × 3 kolom
- **Fitur**:
  - `User-ID`: ID unik pengguna.
  - `Location`: Lokasi pengguna.
  - `Age`: Umur pengguna.
- **Missing Value**:
  - Kolom `Age` memiliki banyak nilai kosong dan outlier (< 5 atau > 100).

### 3. Ratings.csv
- **Dimensi awal**: 1,149,780 baris × 3 kolom
- **Fitur**:
  - `User-ID`: ID pengguna yang memberi rating.
  - `ISBN`: ISBN buku yang diberi rating.
  - `Book-Rating`: Nilai rating dari pengguna (skala 0–10).
- **Missing Value**: Tidak ditemukan missing value.

---

## Data Preparation

Langkah-langkah persiapan data meliputi:

1. **Menghapus missing value**:
   - Menggunakan `.dropna()` untuk kolom penting pada dataset `books`, `users`, dan `ratings`.

2. **Validasi kolom `Year-Of-Publication`**:
   - Mengonversi kolom ke numerik dengan `pd.to_numeric(errors='coerce')`.
   - Filter tahun di luar rentang 1900–2025 dihapus.

3. **Filter usia pengguna**:
   - Hanya mempertahankan usia dalam rentang 5–100.

4. **Threshold aktivitas pengguna dan buku**:
   - Hanya mempertahankan:
     - Pengguna yang memberi ≥ 10 rating.
     - Buku yang menerima ≥ 10 rating.

5. **Konversi tipe data**:
   - Kolom `Book-Rating` dikonversi ke `float32`.

6. **Normalisasi**:
   - Menggunakan `MinMaxScaler()` dari Scikit-learn untuk menormalkan nilai rating ke rentang [0, 1].

7. **Merge data**:
   - Dataset `ratings_filtered` digabung dengan `books_clean` berdasarkan kolom `ISBN`.

---

## Modeling & Results

### Arsitektur Model
Model dibangun menggunakan TensorFlow Keras sebagai berikut:

- **Input**:
  - `User Input`: Embedding layer untuk user ID (panjang embedding: 50)
  - `Book Input`: Embedding layer untuk ISBN (panjang embedding: 50)

- **Hidden Layer**:
  - Concatenate dua embedding
  - Dense layer 128 unit (ReLU)
  - Dense layer 64 unit (ReLU)
  - Dropout 0.2

- **Output**:
  - Dense layer 1 unit (sigmoid) untuk prediksi rating ter-normalisasi

- **Optimasi**:
  - Optimizer: Adam
  - Loss: MSE
  - Epochs: 10, batch size: 128

### Top-5 Rekomendasi Buku

Contoh hasil rekomendasi dari model untuk `User-ID = 276729`:

| Judul Buku                                          | ISBN          | Rating Prediksi |
|-----------------------------------------------------|---------------|------------------|
| Harry Potter and the Prisoner of Azkaban            | 0439136350    | 0.9621           |
| Harry Potter and the Chamber of Secrets             | 0439064864    | 0.9604           |
| Harry Potter and the Sorcerer's Stone               | 059035342X    | 0.9587           |
| The Hobbit                                          | 0345339681    | 0.9552           |
| The Fellowship of the Ring                          | 0618346252    | 0.9510           |

---

## Evaluation

### Metrik Evaluasi:
- **Root Mean Squared Error (RMSE)**: 3.4179  
- **Mean Absolute Error (MAE)**: 2.8303

### Interpretasi:
- RMSE dan MAE menunjukkan bahwa prediksi model memiliki rata-rata kesalahan sekitar 2–3 poin dalam skala rating 0–10. Meskipun masih ada ruang untuk perbaikan, hasil ini cukup baik mengingat dataset asli sangat sparsity dan tidak dilakukan fine-tuning mendalam.

---

## Conclusion

Berdasarkan hasil eksperimen dan evaluasi:

1. **Sistem rekomendasi berhasil dibangun** menggunakan pendekatan collaborative filtering berbasis deep learning dengan embedding neural network.

2. **Model mampu mempelajari preferensi pengguna** dan menghasilkan rekomendasi buku yang sesuai, ditunjukkan oleh output Top-5 rekomendasi yang relevan.

3. **Prediksi rating cukup akurat** (RMSE: 3.42, MAE: 2.79), meskipun performanya masih dapat ditingkatkan dengan lebih banyak data, fitur tambahan, atau tuning model lebih lanjut.

4. **Problem statement telah tercapai**, sistem berhasil:
   - Memberikan rekomendasi yang dipersonalisasi.
   - Memanfaatkan data historis untuk prediksi.
   - Memberikan nilai tambah bagi pengguna.

---
