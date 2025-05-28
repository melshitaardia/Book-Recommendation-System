# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Project Overview

Pada proyek ini, kita akan membangun sistem rekomendasi buku menggunakan pendekatan *Collaborative Filtering*. Dataset yang digunakan adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Ratings.csv) dari Kaggle yang berisi informasi buku, pengguna, dan rating yang diberikan pengguna terhadap buku.

Tujuan utama dari proyek ini adalah merekomendasikan buku berdasarkan riwayat interaksi pengguna sebelumnya.

## Business Understanding

Kembangkan sebuah sistem rekomendasi buku untuk menjawab permasalahan berikut:

### Problem Statements
- Bagaimana membangun sistem rekomendasi buku yang dipersonalisasi menggunakan teknik collaborative filtering berbasis deep learning?
- Bagaimana sistem dapat memprediksi dan merekomendasikan buku lain yang kemungkinan besar disukai pengguna berdasarkan data rating sebelumnya?
- Bagaimana sistem ini dapat memberikan nilai tambah bagi pengguna dalam menemukan buku-buku yang belum pernah mereka baca sebelumnya?

### Goals
- Menghasilkan rekomendasi buku yang dipersonalisasi untuk setiap pengguna berdasarkan pola interaksi dan rating historis.
- Membangun model rekomendasi menggunakan pendekatan embedding neural network untuk collaborative filtering.
- Menyediakan prediksi rating yang akurat sebagai dasar pengambilan keputusan dalam sistem rekomendasi.

## Data Understanding

Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) dan terdiri dari tiga file utama:
- **Books.csv**: berisi informasi seperti ISBN, judul buku, dan penulis.
- **Users.csv**: berisi informasi pengguna seperti user ID dan lokasi.
- **Ratings.csv**: berisi rating yang diberikan oleh pengguna terhadap buku.

Setelah melakukan eksplorasi awal, diketahui bahwa:
- Total **271,379 rating** diberikan oleh **90,000+ pengguna** untuk **140,000+ buku**.
- Terdapat sparsity yang tinggi, sehingga filtering dilakukan untuk mempertahankan pengguna yang memberikan setidaknya 200 rating, dan buku yang menerima setidaknya 100 rating.

## Data Preparation

Tahapan data preparation yang dilakukan:
1. **Pembersihan Data**:
   - Menghapus data duplikat.
   - Menyaring rating yang valid (1 hingga 10).
2. **Filtering**:
   - Mengambil subset data berdasarkan jumlah interaksi minimum (threshold user dan buku).
3. **Mapping**:
   - Mengonversi user ID dan ISBN menjadi indeks integer.
4. **Split Data**:
   - Membagi data menjadi data latih dan data validasi (80:20).

## Modeling

Model yang digunakan adalah **embedding-based collaborative filtering neural network** menggunakan TensorFlow.

Arsitektur model:
- Dua layer embedding: satu untuk pengguna, satu untuk buku.
- Output adalah prediksi rating (float).
- Aktivasi akhir menggunakan Dense linear layer.
- Optimizer: Adam
- Loss: Mean Squared Error

### Training
Model dilatih selama beberapa epoch dan menghasilkan loss yang stabil dan menurun pada training dan validation set.

## Evaluation

### Metrik Evaluasi:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

Hasil evaluasi:
- **RMSE**: 1.32
- **MAE**: 1.01

Metrik ini menunjukkan performa yang cukup baik dalam memprediksi rating pengguna.

## Contoh Top-5 Rekomendasi Buku

| User ID | Top 5 Rekomendasi Buku (ISBN) |
|---------|-------------------------------|
| 250     | ['0446520802', '0316666343', '0679781587', '0671027360', '0312195516'] |
| 114     | ['0316666343', '0446520802', '0312195516', '0679781587', '0671027360'] |
| 8       | ['0446520802', '0316666343', '0312195516', '0679781587', '0671027360'] |

## Catatan

- Sistem ini dapat dikembangkan lebih lanjut dengan memperhatikan konteks konten buku (*Content-Based Filtering*), atau menggabungkannya menjadi *hybrid system*.
- Model juga bisa ditingkatkan dengan eksplorasi teknik regularisasi tambahan seperti **dropout**, dan hyperparameter tuning lanjutan.
