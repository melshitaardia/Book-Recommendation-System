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
  - ISBN: Nomor unik buku.
  - Book-Title: Judul buku.
  - Book-Author: Nama penulis.
  - Year-Of-Publication: Tahun terbit buku.
  - Publisher: Nama penerbit buku.
  - Image-URL-S, Image-URL-M, Image-URL-L: URL gambar buku (beragam ukuran).
- **Missing Values**:
  - Book-Author, Publisher, dan Image-URL-L memiliki nilai kosong.

### 2. Users.csv
- **Dimensi awal**: 278,858 baris × 3 kolom
- **Fitur**:
  - User-ID: ID unik pengguna.
  - Location: Lokasi pengguna.
  - Age: Umur pengguna.
- **Missing Values**:
  - Banyak nilai kosong dan outlier pada kolom Age (< 5 atau > 100).

### 3. Ratings.csv
- **Dimensi awal**: 1,149,780 baris × 3 kolom
- **Fitur**:
  - User-ID: ID pengguna.
  - ISBN: ISBN buku.
  - Book-Rating: Nilai rating pengguna (0–10).
- **Missing Values**:
  - Tidak ditemukan nilai kosong.

---

## Data Preparation

Langkah-langkah yang dilakukan untuk mempersiapkan data:

1. **Transformasi Tahun Terbit**:
   - Kolom `Year-Of-Publication` dikonversi ke tipe numerik dengan `pd.to_numeric(errors='coerce')`.

2. **Filter Tahun Buku**:
   - Hanya mempertahankan buku dengan tahun terbit valid antara **1800 hingga 2025**.

3. **Hapus Missing Value**:
   - `books`: Menghapus baris dengan nilai kosong pada kolom `Book-Author`, `Publisher`, dan `Image-URL-L`.
   - `users`: Menghapus baris dengan nilai kosong pada kolom `Age`.

4. **Filter Usia Pengguna**:
   - Menyaring pengguna dengan `Age` dalam rentang **5–100 tahun**.

5. **Filter Aktivitas Minimum**:
   - Hanya mempertahankan:
     - Pengguna yang memberikan ≥10 rating.
     - Buku yang menerima ≥10 rating.

6. **Normalisasi Rating**:
   - Kolom `Book-Rating` dinormalisasi ke rentang **0–1** menggunakan `MinMaxScaler()` dari scikit-learn.

7. **Gabungkan Dataset**:
   - Dataset rating digabungkan dengan `books_clean` berdasarkan kolom ISBN untuk memberikan konteks tambahan.

8. **Konversi Tipe Data**:
   - Kolom `Book-Rating` dikonversi menjadi `float32` untuk efisiensi memori dan komputasi.

---

## Modeling & Results

### Arsitektur Model

Model dibangun menggunakan TensorFlow Keras dengan konfigurasi sebagai berikut:

- **Input Layer**:
  - User Input: Embedding untuk `user_id`, dimensi **32**.
  - Book Input: Embedding untuk `ISBN`, dimensi **32**.

- **Hidden Layers**:
  - Dua embedding digabungkan (concatenate).
  - Dense layer dengan **64 unit** dan aktivasi ReLU, ditambah regularisasi **L2(0.01)**.
  - Dense layer dengan **32 unit** dan aktivasi ReLU, diikuti oleh **Dropout 0.4**.

- **Output Layer**:
  - Dense 1 unit dengan aktivasi **sigmoid** untuk memprediksi rating dalam skala 0–1.

- **Training Configuration**:
  - Loss Function: **Mean Squared Error (MSE)**
  - Optimizer: **Adam**
  - Epochs: **40**
  - Batch size: **64**

### Hasil Evaluasi

- **Root Mean Squared Error (RMSE)**: **3.4196**
- **Mean Absolute Error (MAE)**: **2.7997**

Hasil ini menunjukkan bahwa model memiliki tingkat kesalahan yang masih dalam batas wajar dan cukup baik untuk baseline sistem rekomendasi.

### Top-10 Rekomendasi Buku untuk User-ID = 276798

Berikut adalah hasil prediksi sistem rekomendasi untuk **User-ID: 276798** berdasarkan model deep learning *collaborative filtering* yang telah dibangun. Sistem menghasilkan 10 buku dengan prediksi rating tertinggi:

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

Prediksi ini berasal dari model yang dilatih pada rating yang dinormalisasi (0–1), kemudian dikalikan kembali ke skala 0–10. Nilai prediksi bisa melebihi batas maksimal karena regresi tidak secara eksplisit membatasi output, namun tetap merefleksikan preferensi relatif pengguna.
---

## Evaluation
Model dilatih menggunakan split data 80:20 (training:test), dan metrik evaluasi dihitung pada data test set untuk mengukur generalisasi model.

### Metrik Evaluasi
- **RMSE (Root Mean Squared Error)**: **3.4196**
- **MAE (Mean Absolute Error)**: **2.7997**

### Interpretasi
Nilai RMSE dan MAE mengindikasikan bahwa model memiliki rata-rata kesalahan sekitar 2–3 poin dari skala rating 0–10. Hasil ini cukup kompetitif untuk baseline sistem rekomendasi berbasis deep learning dan dapat ditingkatkan melalui teknik fine-tuning, penggunaan metadata, atau pendekatan hybrid.

---

## Conclusion

1. **Sistem rekomendasi berhasil dibangun** menggunakan pendekatan collaborative filtering berbasis embedding neural network.
2. **Model menunjukkan performa yang memadai**, dengan RMSE 3.4196 dan MAE 2.7997.
3. **Rekomendasi yang dihasilkan dipersonalisasi** berdasarkan interaksi historis pengguna dan buku.
4. **Sistem membantu pengguna menemukan bacaan baru** tanpa eksplorasi manual.
5. **Problem Statement berhasil dijawab**:
   - Sistem dapat memprediksi dan merekomendasikan buku yang relevan.
   - Memberikan nilai tambah dalam menjelajahi buku baru.
6. **Potensi pengembangan selanjutnya**:
   - Integrasi metadata buku (judul, genre, penulis).
   - Penerapan model hybrid (collaborative + content-based).
   - Eksperimen dengan model advanced seperti Neural Collaborative Filtering (NCF).
