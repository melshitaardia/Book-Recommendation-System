
# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai platform digital, termasuk e-commerce, layanan streaming, dan perpustakaan digital. Seiring dengan pertumbuhan eksponensial data dan banyaknya pilihan yang tersedia, pengguna menghadapi tantangan dalam menemukan item yang sesuai dengan minat mereka secara efisien. Dalam konteks literasi digital, sistem rekomendasi buku memiliki peran strategis untuk membantu pembaca menemukan buku yang relevan dengan preferensi mereka tanpa harus mencarinya secara manual dari ribuan judul yang tersedia.

Pada proyek ini, dikembangkan sebuah sistem rekomendasi buku berbasis *Collaborative Filtering* dengan pendekatan deep learning. Metode ini memanfaatkan interaksi historis antara pengguna dan item (user–book) untuk memprediksi item yang kemungkinan besar disukai oleh pengguna, tanpa memerlukan informasi konten dari buku tersebut.

Berdasarkan studi Bobadilla et al. (2013), *collaborative filtering* telah terbukti menjadi salah satu pendekatan paling efektif dalam sistem rekomendasi karena kemampuannya memberikan personalisasi tinggi dan relevansi yang baik tanpa bergantung pada informasi konten secara eksplisit. Sementara itu, pendekatan deep learning, khususnya model embedding, telah menunjukkan efektivitas dalam menangani masalah *data sparsity* dan skala besar yang umum dalam sistem rekomendasi (Zhang et al., 2019).

Sistem rekomendasi telah banyak diadopsi dalam dunia industri. Misalnya, Amazon menyatakan bahwa lebih dari 35% penjualannya berasal dari sistem rekomendasi (Smith, 2017), dan Goodreads memanfaatkannya untuk membantu pembaca menemukan buku baru.

Dataset yang digunakan dalam proyek ini adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) dari Kaggle, yang mencakup informasi pengguna, ISBN buku, serta rating yang diberikan. Tujuan utama proyek ini adalah membangun model rekomendasi yang mampu memprediksi dan menyarankan buku-buku berdasarkan pola interaksi pengguna dengan tingkat akurasi yang tinggi.

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

Proyek ini menggunakan dataset Book-Crossing yang terdiri dari tiga file utama: `Books.csv`, `Users.csv`, dan `Ratings.csv`. Berikut adalah penjelasan data awal sebelum dilakukan pembersihan.

### 1. Books.csv

- **Dimensi awal**: 271,360 baris × 8 kolom
- **Deskripsi fitur**:
  - `ISBN`: Nomor unik buku.
  - `Book-Title`: Judul buku.
  - `Book-Author`: Nama penulis.
  - `Year-Of-Publication`: Tahun terbit.
  - `Publisher`: Nama penerbit.
  - `Image-URL-S/M/L`: URL sampul buku ukuran kecil/sedang/besar.
- **Masalah**:
  - Kolom `Book-Author`, `Publisher`, dan `Image-URL-L` mengandung missing values.
  - `Year-Of-Publication` mengandung entri tidak valid seperti `'DK Publishing Inc'` dan `'Gallimard'`.

### 2. Users.csv

- **Dimensi awal**: 278,858 baris × 3 kolom
- **Deskripsi fitur**:
  - `User-ID`, `Location`, `Age`
- **Masalah**:
  - Kolom `Age` memiliki nilai kosong dan outlier (di bawah 5 atau di atas 100).

### 3. Ratings.csv

- **Dimensi awal**: 1,149,780 baris × 3 kolom
- **Deskripsi fitur**:
  - `User-ID`, `ISBN`, `Book-Rating`
- **Masalah**:
  - Tidak ditemukan missing values.

---

## Data Preparation

Langkah-langkah persiapan data sebelum modeling:

1. **Menghapus missing values** dari semua dataset dengan `.dropna()`.
2. **Validasi dan pembersihan `Year-Of-Publication`**: non-numerik dikonversi ke NaN dan dibuang jika di luar rentang 1900–2025.
3. **Filter umur** pengguna di luar rentang 5–100 tahun.
4. **Filter pengguna dan buku**: hanya yang memiliki ≥10 interaksi.
5. **Duplikat**: tidak ditemukan duplikat identik, namun tetap disiapkan penghapusan jika muncul.
6. **Konversi tipe data**: `Book-Rating` menjadi `float32`.
7. **Normalisasi rating** dengan `MinMaxScaler` ke rentang 0–1.
8. **Merge dataset**: `ratings_filtered` digabung dengan `books_clean` berdasarkan `ISBN`.

---

## Modeling & Results

Model yang digunakan adalah *Collaborative Filtering* berbasis *Deep Learning Embedding*:

- Dua embedding layer (user dan book) dengan dimensi 50.
- Dua hidden layer dengan aktivasi **ReLU**.
- Output layer dengan aktivasi **linear**.

Model dilatih menggunakan MSE loss dan Adam optimizer, lalu dievaluasi dengan RMSE dan MAE.

### Hasil Evaluasi

- **RMSE**: 3.4179  
- **MAE**: 2.8303  

Model cukup baik dalam mengenali pola, meskipun masih bisa ditingkatkan.

### Top-10 Rekomendasi untuk User-ID 276798

| Judul Buku | ISBN | Prediksi Rating |
|------------|------|------------------|
| *Harry Potter and the Chamber of Secrets Postcard Book* | 0439425220 | 4.68 |
| *There's Treasure Everywhere--A Calvin and Hobbes Collection* | 0836213122 | 3.80 |
| *Die unendliche Geschichte: Von A bis Z* | 3522128001 | 3.68 |
| *Unknown Title* | 0091842050 | 3.62 |
| *Hope for the Flowers* | 0809117541 | 3.62 |
| *Blood Debt (Daw Book Collectors)* | 0886777399 | 3.61 |
| *The Grasshopper Trap* | 0805001115 | 3.58 |
| *UN Viejo Que Leia Novelas De Amor / The Old Man Who Read Love Stories* | 8472236552 | 3.49 |
| *Blind Assassin* | 0747549370 | 3.46 |
| *Romeo and Juliet (Bantam Classic)* | 0553213059 | 3.45 |

---


## Evaluation & Conclusion

### Evaluasi

- **RMSE**: 3.4224
- **MAE**: 2.7774

Model menunjukkan akurasi moderat yang dapat ditingkatkan dengan tuning lebih lanjut.

### Kesimpulan

- Sistem rekomendasi **berhasil mempersonalisasi** saran berdasarkan interaksi historis pengguna.
- Prediksi cukup akurat dalam **memperkirakan preferensi** pengguna.
- Sistem **memberikan nilai tambah** dengan menyarankan buku yang relevan dan belum pernah dibaca.

Dengan demikian, seluruh **problem statement telah terpenuhi**, baik dalam aspek personalisasi maupun pengalaman pengguna.

