# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## 1. Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai platform digital, termasuk e-commerce, layanan streaming, dan perpustakaan digital. Seiring dengan pertumbuhan eksponensial data dan pilihan yang tersedia, pengguna menghadapi kesulitan dalam menemukan item yang sesuai dengan minat mereka secara efisien. Dalam konteks literasi digital, sistem rekomendasi buku memiliki peran strategis untuk membantu pembaca menemukan buku yang relevan dengan preferensi mereka, tanpa harus mencarinya secara manual dari ribuan judul yang tersedia.

Pada proyek ini, dibangun sebuah sistem rekomendasi buku berbasis *Collaborative Filtering* menggunakan pendekatan deep learning. Metode ini memanfaatkan interaksi historis pengguna–item (user–book) untuk memprediksi item yang kemungkinan besar akan disukai oleh pengguna, tanpa perlu mengetahui konten dari buku tersebut.

Berdasarkan studi oleh Bobadilla et al. (2013), *collaborative filtering* telah terbukti sebagai salah satu pendekatan paling efektif dalam membangun sistem rekomendasi karena dapat menghasilkan personalisasi tinggi dan relevansi yang baik meskipun tanpa informasi konten secara eksplisit. Di sisi lain, pendekatan deep learning, khususnya model embedding, telah membuka jalan untuk menangani masalah *data sparsity* dan skala besar yang umum dalam sistem rekomendasi (Zhang et al., 2019).

Sistem rekomendasi telah banyak diadopsi dalam dunia industri. Misalnya, Amazon menyatakan bahwa lebih dari 35% penjualannya berasal dari sistem rekomendasi (Smith, 2017), dan Goodreads memanfaatkan rekomendasi untuk membantu pembaca menemukan buku baru.

Dataset yang digunakan dalam proyek ini adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) dari Kaggle, yang mencakup informasi pengguna, ISBN buku, serta rating yang diberikan. Tujuan utama proyek ini adalah membangun model rekomendasi yang mampu memprediksi dan merekomendasikan buku-buku berdasarkan pola interaksi pengguna sebelumnya dengan tingkat akurasi yang tinggi.

### Referensi

- Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013). Recommender systems survey. *Knowledge-Based Systems*, 46, 109–132. https://doi.org/10.1016/j.knosys.2013.03.012  
- Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. *ACM Computing Surveys (CSUR)*, 52(1), 1–38. https://doi.org/10.1145/3285029  
- Smith, A. (2017). *Amazon Recommendation Engine*. Forbes Tech. https://www.forbes.com/sites/forbestechcouncil/2017/10/16/how-amazons-recommendation-engine-works/

---

## 2. Business Understanding

### Problem Statements
- Bagaimana membangun sistem rekomendasi buku yang dipersonalisasi menggunakan teknik collaborative filtering berbasis deep learning?
- Bagaimana sistem dapat memprediksi dan merekomendasikan buku lain yang kemungkinan besar disukai pengguna berdasarkan data rating sebelumnya?
- Bagaimana sistem ini dapat memberikan nilai tambah bagi pengguna dalam menemukan buku-buku yang belum pernah mereka baca sebelumnya?

### Goals
- Menghasilkan rekomendasi buku yang dipersonalisasi untuk setiap pengguna berdasarkan pola interaksi dan rating historis.
- Membangun model rekomendasi menggunakan pendekatan embedding neural network untuk collaborative filtering.
- Menyediakan prediksi rating yang akurat sebagai dasar pengambilan keputusan dalam sistem rekomendasi.

### Solution Approach

1. **Collaborative Filtering (yang digunakan)**:
   Mengandalkan pola interaksi historis pengguna dengan buku (user–item matrix) menggunakan embedding-based neural network.

2. **Content-Based Filtering (alternatif)**:
   Menggunakan metadata buku seperti judul, penulis, atau kategori, model dapat mempelajari kesamaan antar buku dan merekomendasikan buku serupa dengan yang disukai pengguna sebelumnya.

---

## 3. Data Understanding

Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) dan terdiri dari tiga file utama:
- **Books.csv**: berisi informasi seperti ISBN, judul buku, dan penulis.
- **Users.csv**: berisi informasi pengguna seperti user ID dan lokasi.
- **Ratings.csv**: berisi rating yang diberikan oleh pengguna terhadap buku.

### Informasi Umum:
- Total **271,379 rating** diberikan oleh **90,000+ pengguna** untuk **140,000+ buku**.
- Terdapat sparsity yang tinggi (banyak kombinasi user-book tidak memiliki rating).

### Visualisasi & Insight:
- **Distribusi Rating**: Mayoritas berada pada rentang 6–9.
- **Distribusi Rating per User**: Banyak pengguna hanya memberi sedikit rating.
- **Distribusi Rating per Buku**: Beberapa buku populer memiliki ratusan rating, sementara sebagian besar hanya memiliki satu.

Visualisasi yang digunakan: histogram rating, bar chart top-rated books, sebaran rating per user/buku.

---

## 4. Data Preparation

Langkah-langkah yang dilakukan:
1. **Pembersihan Data**:
   - Menghapus entri duplikat.
   - Menyaring rating yang valid (1–10).

2. **Filtering**:
   - Mengambil subset data berdasarkan jumlah interaksi minimum.
   - Pengguna yang memberikan ≥ 200 rating dan buku dengan ≥ 100 rating dipertahankan.

3. **Mapping**:
   - User-ID dan ISBN dikonversi menjadi integer index menggunakan `LabelEncoder`.

4. **Split Data**:
   - 80% data untuk training, 20% untuk validation.
   - Rating dinormalisasi ke skala 0–1 untuk pembelajaran neural network.

---

## 5. Modeling & Results

### Model Utama: Collaborative Filtering with Neural Network

#### Arsitektur:
- **User Embedding Layer**: Mengubah user ID menjadi vektor.
- **Book Embedding Layer**: Mengubah ISBN menjadi vektor.
- Kedua embedding digabung dan diteruskan ke Dense Layer → Output rating.

#### Konfigurasi:
- Optimizer: Adam
- Loss: Mean Squared Error
- Aktivasi akhir: Linear
- Epoch: 20
- Batch Size: 256

### Hasil Prediksi Top-10 Buku untuk `User-ID: 276762`

| Judul Buku                                                                 | ISBN         | Prediksi Rating |
|---------------------------------------------------------------------------|--------------|------------------|
| All-American Girl                                                         | 0060294698   | 5.24             |
| Love You Forever                                                          | 0920668372   | 5.05             |
| The Lorax                                                                 | 0394823370   | 5.02             |
| The Night the Bear Ate Goombaw                                           | 0805013407   | 4.97             |
| The Watchers Guide Buffy The Vampire Slayer (Buffy The Vampire Slayer)   | 0671024337   | 4.79             |
| Deep Thoughts                                                             | 0425133656   | 4.76             |
| Unknown Title                                                             | 3596215226   | 4.74             |
| Stardust of Yesterday (Haunted Hearts)                                   | 0515118397   | 4.72             |
| Something Wicked This Way Comes                                          | 0380729407   | 4.70             |
| TOLKIEN MAGNETIC POSTCARDS(tm) 12 Full-color Magnetic Postcards to Send or Save | 0762409533   | 4.68             |

---

### Alternatif Modeling: Matrix Factorization (SVD)

Model pembanding dibangun menggunakan pendekatan matrix factorization berbasis SVD dari pustaka `Surprise`.

#### Hasil Evaluasi:
- RMSE: 1.39
- MAE: 1.11

Model neural network memberikan hasil yang lebih baik dan fleksibel untuk data besar.

---

## 6. Evaluation

### Evaluation Metrics:

- **Root Mean Squared Error (RMSE)**:
  
  \[
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2}
  \]

- **Mean Absolute Error (MAE)**:
  
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{r}_i - r_i|
  \]

### Hasil Evaluasi Model:
- RMSE: **1.32**
- MAE: **1.01**

Angka ini menunjukkan model mampu melakukan prediksi rating cukup akurat dengan error relatif rendah.

---

## 7. Catatan & Pengembangan Lanjutan

- Model dapat dikembangkan lebih lanjut dengan menambahkan informasi konten (judul, kategori, sinopsis) untuk membentuk sistem **hybrid recommendation**.
- Teknik **regularisasi** seperti dropout dan early stopping dapat digunakan untuk meningkatkan generalisasi.
- Perlu eksplorasi terhadap **implicit feedback** (click, view) dan bukan hanya explicit rating.

---

## 8. Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi buku berbasis collaborative filtering dengan pendekatan deep learning. Model mampu memberikan rekomendasi yang cukup personal dan akurat, serta dapat diperluas untuk mendukung rekomendasi berbasis konten.
