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

**Sumber Dataset**:  
Dataset ini diambil dari Kaggle: [https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) 

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

9. **Encoding ID Unik**:
   - Kolom `User-ID` dan `ISBN` tidak bisa langsung digunakan dalam model, sehingga dikodekan (encoded) menjadi indeks numerik menggunakan dictionary mapping.
   - Dua dictionary digunakan untuk mengubah ID asli ke indeks numerik:
     - `user2user_encoded`: untuk memetakan `User-ID` ke indeks integer.
     - `book2book_encoded`: untuk memetakan `ISBN` ke indeks integer.
   - Sebaliknya, dictionary `user_encoded2user` dan `book_encoded2book` digunakan untuk mengembalikan indeks ke ID asli saat interpretasi hasil rekomendasi.

10. **Pemisahan Dataset (train_test_split)**:
   - Dataset hasil preprocessing dibagi menjadi data pelatihan dan validasi dengan fungsi `train_test_split` dari `sklearn.model_selection`.
   - Proporsi data adalah **80% untuk pelatihan** dan **20% untuk validasi**, dengan parameter `random_state` untuk menjaga reprodusibilitas hasil.
      
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

| No. | Judul Buku                                                                  | ISBN         | Prediksi Rating |
|-----|-----------------------------------------------------------------------------|--------------|------------------|
| 1   | Saving Grace                                                                | 0345403339   | **5.08**         |
| 2   | Living Juicy: Daily Morsels for Your Creative Soul                          | 0890877033   | **4.84**         |
| 3   | Odyssey: The Story of Odysseus                                              | 0451628055   | **4.73**         |
| 4   | FAT!SO? : Because You Don't Have to Apologize for Your Size                | 0898159954   | **4.71**         |
| 5   | Cows Of Our Planet (Far Side Series)                                        | 0836217012   | **4.69**         |
| 6   | Hotel of the Saints                                                         | 0684843102   | **4.63**         |
| 7   | The Giving Tree                                                             | 0060256656   | **4.63**         |
| 8   | Almost blue (Stile libero)                                                  | 8806143042   | **4.59**         |
| 9   | I Just Forgot (A Little Critter Book)                                       | 0307119750   | **4.57**         |
| 10  | It's Not About the Bike: My Journey Back to Life                            | 0399146113   | **4.55**         |

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
