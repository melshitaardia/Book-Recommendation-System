# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Project Overview

Sistem rekomendasi telah menjadi elemen penting dalam banyak aplikasi digital, termasuk platform pembelian buku online. Banyaknya pilihan buku membuat pengguna memerlukan bantuan untuk menemukan buku yang relevan dengan minat mereka.

Proyek ini bertujuan untuk membangun **sistem rekomendasi buku yang dipersonalisasi** berdasarkan data rating dan metadata dari pengguna dan buku, menggunakan pendekatan *collaborative filtering berbasis deep learning*.

Sistem rekomendasi ini dibangun dengan memanfaatkan dataset Book-Crossing yang memuat interaksi antara pengguna dan buku. Model yang digunakan didesain untuk mempelajari representasi (embedding) pengguna dan buku, sehingga mampu merekomendasikan buku yang relevan dan belum pernah dibaca oleh pengguna tersebut.

## Business Understanding

### Problem Statements

- Bagaimana membangun sistem rekomendasi buku berdasarkan interaksi pengguna sebelumnya?
- Bagaimana sistem dapat memberikan rekomendasi yang relevan untuk pengguna yang belum pernah memberi rating pada banyak buku?
- Bagaimana membuat sistem rekomendasi yang tidak hanya mengandalkan popularitas umum, tetapi memperhatikan preferensi unik tiap pengguna?

### Goals

- Menghasilkan rekomendasi buku yang sesuai dengan preferensi pengguna berdasarkan rating sebelumnya.
- Membangun model rekomendasi dengan pendekatan *collaborative filtering* yang mampu menghasilkan **top-N recommendation** untuk pengguna.
- Menyediakan sistem yang dapat membantu pengguna menemukan buku yang belum pernah mereka eksplorasi sebelumnya.

### Solution Approach

#### Solution Statements
- **Collaborative Filtering berbasis Deep Learning**: menggunakan arsitektur neural network embedding untuk memahami hubungan antara pengguna dan buku berdasarkan data rating.
- (Opsional untuk perluasan): **Content-based Filtering** jika ingin memanfaatkan metadata buku seperti genre atau penulis di masa mendatang.

## Data Understanding

Dataset yang digunakan adalah [Book-Crossing Dataset (BX-Books, BX-Users, BX-Ratings)](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset), yang terdiri dari tiga komponen:
- `BX-Books.csv`: informasi metadata buku seperti ISBN, judul, dan penulis.
- `BX-Users.csv`: informasi pengguna seperti user ID, lokasi, dan usia.
- `BX-Ratings.csv`: rating eksplisit yang diberikan pengguna terhadap buku.

### Jumlah Data
- Jumlah rating: 1.149.780 entri
- Jumlah buku: 271.379
- Jumlah pengguna: 278.858

### Fitur Penting
- `user_id`: ID pengguna
- `isbn`: ID buku
- `book_title`: Judul buku
- `book_author`: Nama penulis
- `location`: Lokasi pengguna
- `age`: Usia pengguna
- `rating`: Rating dari pengguna terhadap buku (skala 1–10)

## Data Preparation

Langkah-langkah preprocessing dan data preparation yang dilakukan:

1. **Pembersihan data duplikat**: menghapus entri duplikat pada BX-Users dan BX-Books.
2. **Pembersihan rating 0**: hanya mempertahankan rating eksplisit (> 0).
3. **Filtering pengguna dan buku**: hanya mempertahankan pengguna yang memberikan ≥10 rating, dan buku yang menerima ≥5 rating.
4. **Label encoding user_id dan isbn**: mengubah ke bentuk numerik untuk keperluan embedding.
5. **Normalisasi rating**: rating dinormalisasi ke rentang [0, 1] untuk digunakan sebagai target dalam training.

## Modeling

Model rekomendasi dibangun menggunakan pendekatan **Collaborative Filtering dengan Neural Network**:

- Menggunakan embedding layer untuk `user_id` dan `book_id`.
- Menggabungkan embedding dan melewatkannya ke hidden layers.
- Arsitektur:
  - Embedding user dan item masing-masing ukuran 50.
  - 2 hidden layer: Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3)
  - Output layer: Dense(1) (regresi untuk rating)
- Fungsi aktivasi: ReLU
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Metode evaluasi: Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE)

### Top-N Recommendation
Model dilatih dan digunakan untuk menghasilkan rekomendasi **Top-10 buku** untuk user ID `276762`, berdasarkan prediksi tertinggi dari buku yang belum pernah dirating oleh pengguna tersebut.

## Evaluation

Model dievaluasi menggunakan data validasi dan data uji. Metrik evaluasi:

- **Root Mean Squared Error (RMSE)**:
  \[
  RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  \]
- **Mean Absolute Error (MAE)**:
  \[
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]

### Hasil Evaluasi
- RMSE (testing): sekitar **1.02**
- MAE (testing): sekitar **0.82**

Model menunjukkan performa cukup baik untuk sistem rekomendasi berbasis rating skala 1–10.

---

_Catatan:_
- Laporan ini berdasarkan proyek sistem rekomendasi buku dengan dataset publik dari Kaggle, dibangun menggunakan TensorFlow dan scikit-learn.
- Top-N recommendation telah disajikan dan bisa diperluas untuk integrasi ke platform web atau aplikasi pembaca buku.
