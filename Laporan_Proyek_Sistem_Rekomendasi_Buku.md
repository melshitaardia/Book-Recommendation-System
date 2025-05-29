# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai platform digital, termasuk e-commerce, layanan streaming, dan perpustakaan digital. Dalam konteks literasi digital, sistem rekomendasi buku membantu pembaca menemukan buku yang relevan tanpa harus mencarinya secara manual dari ribuan judul yang tersedia.

Pada proyek ini, dikembangkan sebuah sistem rekomendasi buku berbasis *Collaborative Filtering* dengan pendekatan deep learning. Metode ini memanfaatkan interaksi historis antara pengguna dan buku untuk memprediksi item yang kemungkinan besar disukai oleh pengguna, tanpa memerlukan informasi konten eksplisit dari buku tersebut.

Studi Bobadilla et al. (2013) menunjukkan bahwa *collaborative filtering* efektif karena mampu memberikan personalisasi tinggi dan relevansi yang baik. Sementara itu, pendekatan deep learning, khususnya model embedding, menunjukkan efektivitas dalam menangani masalah *data sparsity* dan skala besar yang umum dalam sistem rekomendasi (Zhang et al., 2019).

Dataset yang digunakan adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) dari Kaggle, yang mencakup informasi pengguna, ISBN buku, serta rating yang diberikan.

### Referensi

- Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013). Recommender systems survey. *Knowledge-Based Systems*, 46, 109–132.
- Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. *ACM Computing Surveys (CSUR)*, 52(1), 1–38.
- Smith, A. (2017). *Amazon Recommendation Engine*. Forbes Tech.

---

## Business Understanding

### Problem Statements
- Bagaimana membangun sistem rekomendasi buku yang dipersonalisasi menggunakan teknik collaborative filtering berbasis deep learning?
- Bagaimana sistem dapat memprediksi dan merekomendasikan buku yang kemungkinan besar disukai pengguna berdasarkan data rating sebelumnya?
- Bagaimana sistem ini dapat memberikan nilai tambah bagi pengguna dalam menemukan buku-buku yang belum pernah mereka baca sebelumnya?

### Goals
- Menghasilkan rekomendasi buku yang dipersonalisasi berdasarkan interaksi historis pengguna.
- Membangun model rekomendasi menggunakan embedding neural network.
- Menyediakan prediksi rating yang akurat sebagai dasar pengambilan keputusan.

### Solution Approach
1. **Collaborative Filtering (yang digunakan)**:  
   Mengandalkan pola interaksi pengguna dengan buku (user–item matrix) menggunakan embedding-based neural network.
2. **Content-Based Filtering (alternatif)**:  
   Menggunakan metadata buku seperti judul, penulis, atau kategori untuk merekomendasikan buku serupa.

---

## Data Understanding

Dataset terdiri dari tiga file utama:

### 1. Books.csv
- **Dimensi awal**: 271,360 baris × 8 kolom
- **Fitur**:
  - `ISBN`: Nomor unik buku.
  - `Book-Title`: Judul buku.
  - `Book-Author`: Nama penulis.
  - `Year-Of-Publication`: Tahun terbit buku.
  - `Publisher`: Nama penerbit buku.
  - `Image-URL-S`: URL gambar buku (ukuran kecil).
  - `Image-URL-M`: URL gambar buku (ukuran sedang).
  - `Image-URL-L`: URL gambar buku (ukuran besar).
- **Missing Values**:
  - Kolom `Book-Author`, `Publisher`, dan `Image-URL-L` memiliki nilai kosong sebelum pembersihan.

### 2. Users.csv
- **Dimensi awal**: 278,858 baris × 3 kolom
- **Fitur**:
  - `User-ID`: ID unik pengguna.
  - `Location`: Lokasi pengguna.
  - `Age`: Umur pengguna.
- **Missing Values**:
  - Kolom `Age` memiliki banyak nilai kosong dan outlier (< 5 atau > 100).

### 3. Ratings.csv
- **Dimensi awal**: 1,149,780 baris × 3 kolom
- **Fitur**:
  - `User-ID`: ID pengguna.
  - `ISBN`: ISBN buku.
  - `Book-Rating`: Nilai rating pengguna (0–10).
- **Missing Values**:
  - Tidak ditemukan nilai kosong.

---

## Data Preparation

Langkah-langkah yang dilakukan untuk mempersiapkan data:

1. **Menghapus missing value**:
   - `books`: Drop baris yang memiliki nilai kosong pada `Book-Author`, `Publisher`, dan `Image-URL-L`.
   - `users`: Drop baris dengan nilai kosong pada kolom `Age`.
   - `ratings`: Tidak ada missing value.

2. **Filter tahun terbit buku**:
   - Kolom `Year-Of-Publication` dikonversi menjadi numerik dengan `pd.to_numeric(errors='coerce')`.
   - Hanya mempertahankan buku dengan tahun terbit valid antara 1900 hingga 2025.

3. **Filter usia pengguna**:
   - Hanya mempertahankan pengguna dengan `Age` dalam rentang 5–100.

4. **Konversi tipe data**:
   - Kolom `Book-Rating` dikonversi menjadi `float32` untuk efisiensi komputasi.

5. **Threshold aktivitas**:
   - Hanya mempertahankan:
     - Pengguna yang memberikan ≥10 rating.
     - Buku yang menerima ≥10 rating.

6. **Normalisasi**:
   - Nilai `Book-Rating` dinormalisasi ke skala 0–1 menggunakan `MinMaxScaler()` dari scikit-learn.

7. **Penggabungan data**:
   - Data rating digabung dengan informasi buku (`books_clean`) berdasarkan `ISBN`.

---

## Modeling & Results

### Arsitektur Model

Model dibangun menggunakan TensorFlow Keras dengan arsitektur:

- **Input**:
  - User Input: Embedding untuk `user_id` (dimensi: 50).
  - Book Input: Embedding untuk `ISBN` (dimensi: 50).

- **Hidden Layers**:
  - Concatenate dua embedding.
  - Dense 128 unit (ReLU).
  - Dense 64 unit (ReLU).
  - Dropout 0.2.

- **Output Layer**:
  - Dense 1 unit (sigmoid) untuk memprediksi rating dalam skala 0–1.

- **Kompilasi**:
  - Loss: Mean Squared Error (MSE)
  - Optimizer: Adam
  - Epochs: 10
  - Batch size: 128

### Hasil Evaluasi

- **Root Mean Squared Error (RMSE)**: **3.4227**
- **Mean Absolute Error (MAE)**: **2.7963**

Nilai ini menunjukkan bahwa model memiliki kemampuan prediksi yang cukup baik untuk sistem rekomendasi, meskipun masih dapat ditingkatkan dengan tuning lanjutan.

### Top-10 Rekomendasi Buku untuk User-ID = 276798

| No. | Judul Buku                                                                 | ISBN          | Prediksi Rating |
|-----|----------------------------------------------------------------------------|---------------|------------------|
| 1   | All-American Girl                                                         | 0060294698    | **5.08**         |
| 2   | Die unendliche Geschichte: Von A bis Z                                    | 3522128001    | **4.86**         |
| 3   | The Baby Book: Everything You Need to Know About Your Baby from Birth to Age Two | 0316779059    | **4.75**         |
| 4   | High Exposure: An Enduring Passion for Everest and Unforgiving Places    | 0684865459    | **4.68**         |
| 5   | The Art of Raising a Puppy                                                | 0316578398    | **4.33**         |
| 6   | Thomas and the School Trip (Step-Into-Reading, Step 2)                   | 0679843655    | **4.28**         |
| 7   | The Berenstain Bears' New Baby (Pictureback Series)                      | 0394829085    | **4.25**         |
| 8   | Sex, Drugs, and Cocoa Puffs : A Low Culture Manifesto                    | 0743236009    | **4.18**         |
| 9   | Harry Potter and the Prisoner of Azkaban (Book 3)                        | 0439136350    | **4.14**         |
| 10  | The Watchers Guide Buffy The Vampire Slayer                              | 0671024337    | **4.11**         |

---

## Evaluation

### Metrik Evaluasi
Model dievaluasi dengan:

- **RMSE (Root Mean Squared Error)**: 3.4227
- **MAE (Mean Absolute Error)**: 2.7963

### Interpretasi
Nilai RMSE dan MAE menunjukkan bahwa model memiliki rata-rata kesalahan prediksi rating sekitar 2–3 poin dari skala 0–10. Ini cukup baik untuk tahap prototipe awal, dengan mempertimbangkan keterbatasan data dan tidak dilakukan fine-tuning lanjutan.

---

## Conclusion

Berdasarkan hasil proyek, dapat disimpulkan:

1. **Sistem rekomendasi berhasil dibangun** menggunakan collaborative filtering berbasis embedding neural network.
2. **Model mampu mempelajari preferensi pengguna** dari data historis dan memberikan rekomendasi buku yang dipersonalisasi.
3. **Prediksi rating cukup akurat**, dengan RMSE 3.42 dan MAE 2.79, yang mencerminkan performa memadai sebagai baseline model.
4. **Sistem memberikan nilai tambah** dalam menyederhanakan pencarian buku dan membantu pengguna menemukan bacaan baru yang relevan.
5. **Problem Statement terjawab**:
   - Sistem berhasil memprediksi dan merekomendasikan buku berdasarkan interaksi pengguna.
   - Sistem memberikan nilai tambah dalam menemukan buku baru yang relevan.
6. **Peluang pengembangan**:
   - Menambahkan fitur konten buku.
   - Menerapkan model hybrid atau fine-tuning parameter untuk performa lebih tinggi.

Proyek ini layak dijadikan dasar pengembangan sistem rekomendasi buku lebih lanjut.

---
