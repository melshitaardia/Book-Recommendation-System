
# Laporan Proyek Machine Learning - Melshita Ardia Kirana

## Project Overview

Proyek ini bertujuan untuk melakukan analisis eksploratori data (EDA) pada dataset sistem rekomendasi buku yang mencakup informasi pengguna, ISBN, dan rating. Dengan eksplorasi ini, diharapkan diperoleh pemahaman mendalam terkait pola interaksi pengguna dan buku yang nantinya akan menjadi dasar dalam membangun sistem rekomendasi berbasis machine learning, khususnya dengan pendekatan collaborative filtering berbasis deep learning.

## Business Understanding

Kembangkan sebuah sistem rekomendasi buku untuk menjawab permasalahan berikut:

Berdasarkan data mengenai pengguna dan buku (termasuk ISBN dan rating), bagaimana membangun sistem rekomendasi yang dipersonalisasi dengan teknik collaborative filtering berbasis deep learning?

Dengan data rating yang tersedia, bagaimana perusahaan dapat merekomendasikan buku lain yang mungkin disukai oleh pengguna dan belum pernah mereka baca sebelumnya?

Untuk menjawab pertanyaan tersebut, sistem rekomendasi ini dikembangkan dengan tujuan atau goals sebagai berikut:

- Menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk setiap pengguna berdasarkan pola interaksi mereka dengan buku lain menggunakan pendekatan embedding neural network (collaborative filtering).
- Menyediakan prediksi rating buku yang akurat sebagai dasar pengambilan keputusan rekomendasi.

## Data Understanding

Dataset terdiri dari 3 bagian utama:
- **Books.csv**: berisi informasi buku (ISBN, judul, pengarang, tahun terbit, dan penerbit)
- **Users.csv**: berisi data pengguna (User-ID, lokasi, dan usia)
- **Ratings.csv**: berisi interaksi pengguna dengan buku berupa rating (User-ID, ISBN, Book-Rating)

Dataset ini diambil dari dataset populer Book-Crossing dan digunakan untuk eksplorasi sistem rekomendasi. Terdapat sekitar:
- 271,379 entri rating
- 1.1 juta entri buku
- 278,858 pengguna

## Data Preparation

Beberapa langkah pembersihan dan persiapan data yang dilakukan:

- **Pembersihan kolom kosong dan tidak relevan**, seperti penghapusan `Image-URL` pada dataset `books`.
- **Gabungan tiga dataset** (`ratings`, `books`, `users`) untuk mendapatkan struktur data yang utuh.
- **Menghapus data duplikat** dan **mengisi nilai yang hilang** (jika ada).
- Transformasi teks ke format yang konsisten seperti lowercase, stripping whitespaces, dan lainnya jika diperlukan.

## Modeling

Tahapan modeling **belum dilakukan** dalam proyek ini. Proyek ini masih fokus pada tahap eksplorasi data dan pembersihan dataset sebagai pondasi untuk proses modeling sistem rekomendasi di tahap selanjutnya.

## Evaluation

Karena belum ada tahap modeling, maka belum ada evaluasi performa sistem rekomendasi. Namun, dilakukan analisis eksploratif terhadap data sebagai dasar memahami pola-pola umum yang akan berguna dalam modeling, di antaranya:

### Visualisasi Rating
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
sns.countplot(x='Book-Rating', data=ratings_df)
plt.title('Distribusi Rating Buku')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.show()
```

### Ringkasan Statistik Awal
| Statistik        | Jumlah       |
|------------------|--------------|
| Total Rating     | 271,379      |
| Jumlah Buku Unik | 242,135      |
| Jumlah Pengguna  | 278,858      |
| Rating 0 (implisit) | Mayoritas |
| Rating 5–10 (eksplisit) | Minoritas |

---

_Proyek ini akan dilanjutkan ke tahap modeling dan evaluasi sistem rekomendasi menggunakan teknik collaborative filtering pada iterasi berikutnya._
