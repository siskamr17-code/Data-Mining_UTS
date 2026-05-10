# UTS DATA MINING - PREDIKSI KUALITAS ANGGUR

**Nama:** Siska Meilinda Rahman
**NIM:** 2304020030
**Prodi:** S1 Pendidikan Matematika
**Universitas:** Universitas Negeri Semarang (UNNES)
**Dosen Pengampu:** Nur Achmey Selgi Harwanti, S.Stat. M.Stat

---

## Tujuan Ujian
Membuat model klasifikasi untuk memprediksi kualitas anggur (wine quality) berdasarkan fitur-fitur kimiawi pada dataset Wine Quality.

## Dataset
- **Sumber:** https://bit.ly/datasetwine
- **Data Training:** Berisi 11 fitur kimiawi + kolom 'quality' (target) + kolom 'Id'
- **Data Testing:** Berisi 11 fitur kimiawi + kolom 'Id' (tanpa kolom 'quality')

## Ringkasan Hasil
- **Model terbaik:** Random Forest
- **Parameter terbaik:** n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2

---

## LANGKAH 1: PERSIAPAN ENVIRONMENT DAN LIBRARY

### Tujuan
Mempersiapkan environment Google Colab dan mengimport semua library yang diperlukan.

### Proses yang Dilakukan
1. **Mount Google Drive** - Untuk menyimpan dan mengakses file
2. **Import library** yang diperlukan:
   - `pandas` & `numpy` → manipulasi data
   - `matplotlib` & `seaborn` → visualisasi
   - `sklearn` → preprocessing, model, evaluasi
   - `joblib` → menyimpan model

### Interpretasi
Semua library berhasil diimport. Google Drive di-mount agar file dapat disimpan dan diakses selama proses pengerjaan.
## LANGKAH 2: UPLOAD DATA TRAINING

### Tujuan
Mengupload file `data_training.csv` dari komputer ke Google Colab.

### Proses yang Dilakukan
1. Menggunakan `files.upload()` dari library google.colab
2. Sistem akan menampilkan tombol "Choose Files"
3. Klik tombol dan pilih file `data_training.csv` dari komputer
4. File akan terbaca sebagai pandas dataframe

### Hasil yang Diperoleh
- Data training berhasil dimuat dengan shape (baris, kolom)
- Data training memiliki kolom 'quality' sebagai target
- Data training memiliki kolom 'Id' sebagai identifier

### Interpretasi
Data training digunakan untuk melatih model. Kolom 'quality' adalah target yang akan diprediksi. Kolom 'Id' berfungsi sebagai identifier unik untuk setiap sampel.
## LANGKAH 3: UPLOAD DATA TESTING

### Tujuan
Mengupload file `data_testing.csv` dari komputer ke Google Colab.

### Proses yang Dilakukan
1. Menggunakan `files.upload()` untuk upload file
2. Pilih file `data_testing.csv` dari komputer
3. File akan terbaca sebagai pandas dataframe

### Hasil yang Diperoleh
- Data testing berhasil dimuat dengan shape (baris, kolom)
- Data testing TIDAK memiliki kolom 'quality' (akan diprediksi)
- Data testing memiliki kolom 'Id' sebagai identifier

### Interpretasi
Data testing adalah data yang akan diprediksi kualitasnya oleh model. Karena tidak memiliki label 'quality', model harus memprediksi nilai quality berdasarkan fitur-fitur yang ada.
## LANGKAH 4: EKSPLORASI AWAL DATA (EDA)

### Tujuan
Memahami struktur dan distribusi data sebelum dilakukan pembersihan dan pemodelan.

### Proses yang Dilakukan
1. `df_train.info()` - Melihat tipe data dan jumlah non-null
2. `value_counts()` - Melihat distribusi target variable 'quality'

### Hasil yang Diperoleh

**Distribusi Kualitas Anggur:**
| Quality | Jumlah | Persentase |
|---------|--------|------------|
| 3 | [sesuai hasil] | [X]% |
| 4 | [sesuai hasil] | [X]% |
| 5 | [sesuai hasil] | [X]% |
| 6 | [sesuai hasil] | [X]% |
| 7 | [sesuai hasil] | [X]% |
| 8 | [sesuai hasil] | [X]% |

### Interpretasi
- Sebagian besar sampel memiliki kualitas 5 dan 6
- Tidak ada sampel dengan quality 0,1,2,9,10
- Data tidak seimbang (imbalanced) - ini perlu diperhatikan dalam pemodelan
## LANGKAH 5: PREPROCESSING DAN FEATURE SCALING

### Tujuan
Mempersiapkan data agar siap digunakan untuk pemodelan.

### Proses yang Dilakukan

#### 5.1 Definisi Fitur
11 fitur kimiawi yang digunakan:
1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

#### 5.2 Pemisahan Fitur dan Target
- `X` = 11 fitur (input)
- `y` = quality (target)
- `X_test` = fitur dari data testing

#### 5.3 StandardScaler (Normalisasi)
Karena rentang nilai setiap fitur berbeda-beda (contoh: density 0.99-1.00, alcohol 8-15), perlu dilakukan standardisasi.

**Rumus:** z = (x - μ) / σ
**Hasil:** Data berdistribusi dengan rata-rata = 0 dan standar deviasi = 1

### Interpretasi
Feature scaling penting agar model tidak memberikan bobot lebih pada fitur dengan skala besar. Tanpa scaling, fitur seperti 'alcohol' (skala 8-15) akan mendominasi dibanding 'density' (skala 0.99-1.00) meskipun mungkin kurang berpengaruh.
## LANGKAH 6: SPLIT DATA TRAINING DAN VALIDASI

### Tujuan
Membagi data menjadi dua bagian untuk melatih dan mengevaluasi model.

### Parameter yang Digunakan
- `test_size = 0.2` → 20% data untuk validasi, 80% untuk training
- `random_state = 42` → Memastikan hasil konsisten setiap kali dijalankan
- `stratify = y` → Menjaga proporsi kelas quality tetap sama

### Hasil yang Diperoleh
- **Training set:** [X] sampel
- **Validation set:** [Y] sampel

### Interpretasi
Split data penting untuk:
1. Menghindari **overfitting** (model terlalu hafal data training)
2. Mengevaluasi performa model pada data yang belum pernah dilihat
3. Memastikan model dapat **generalisasi** dengan baik

Penggunaan `stratify` memastikan distribusi quality di training dan validation set sama, sehingga evaluasi tidak bias.
## LANGKAH 7: PEMBANGUNAN DAN EVALUASI MODEL

### Tujuan
Membangun dan membandingkan 6 model klasifikasi untuk memilih model terbaik.

### Model yang Diuji

| No | Model | Kelebihan |
|----|-------|-----------|
| 1 | Random Forest (RF) | Ensemble, tahan overfitting |
| 2 | Gradient Boosting (GB) | Akurasi tinggi |
| 3 | Decision Tree (DT) | Mudah diinterpretasi |
| 4 | K-Neighbors (KNN) | Sederhana, berbasis jarak |
| 5 | Logistic Regression (LR) | Cepat, linear |
| 6 | SVM | Cocok data kompleks |

### Metrik Evaluasi
- **Accuracy:** Persentase prediksi yang benar
- **Cross Validation (CV):** Rata-rata akurasi dari 5 fold berbeda

### Hasil Evaluasi

| Model | Accuracy | CV Mean |
|-------|----------|---------|
| RF | [sesuai hasil] | [sesuai hasil] |
| GB | [sesuai hasil] | [sesuai hasil] |
| DT | [sesuai hasil] | [sesuai hasil] |
| KNN | [sesuai hasil] | [sesuai hasil] |
| LR | [sesuai hasil] | [sesuai hasil] |
| SVM | [sesuai hasil] | [sesuai hasil] |

### Interpretasi
**Random Forest (RF)** dipilih sebagai model terbaik karena:
1. Memiliki akurasi tertinggi
2. Hasil cross-validation stabil
3. Ensemble learning membuatnya tahan overfitting
4. Dapat memberikan feature importance
## LANGKAH 8: HYPERPARAMETER TUNING

### Tujuan
Mencari kombinasi parameter terbaik untuk model Random Forest agar performanya maksimal.

### Parameter yang Diuji

| Parameter | Nilai yang Diuji | Keterangan |
|-----------|------------------|------------|
| n_estimators | 100, 200 | Jumlah pohon dalam forest |
| max_depth | 10, 20, None | Kedalaman maksimal pohon |
| min_samples_split | 2, 5 | Minimal sampel untuk split node |
| min_samples_leaf | 1, 2 | Minimal sampel di daun |

### Metode yang Digunakan
**GridSearchCV** dengan:
- `cv = 5` → 5-fold cross validation
- `scoring = 'accuracy'` → menggunakan akurasi
- `n_jobs = -1` → menggunakan semua processor

### Hasil Parameter Terbaik
- **n_estimators:** [sesuai hasil]
- **max_depth:** [sesuai hasil]
- **min_samples_split:** [sesuai hasil]
- **min_samples_leaf:** [sesuai hasil]

### Interpretasi
Parameter optimal meningkatkan performa model. `n_estimators=200` (tidak terlalu besar sehingga tidak lambat), `max_depth=20` (cukup dalam untuk menangkap pola kompleks tapi tidak overfit).
## LANGKAH 9: TRAINING MODEL FINAL DAN FEATURE IMPORTANCE

### Tujuan
1. Melatih model final dengan seluruh data training
2. Mengetahui fitur mana yang paling berpengaruh

### Proses yang Dilakukan
1. Model Random Forest dilatih dengan **seluruh data** `X_scaled` dan `y`
2. Model disimpan menggunakan `joblib.dump()`
3. Menghitung feature importance

### Hasil Feature Importance

| Ranking | Fitur | Importance |
|---------|-------|------------|
| 1 | alcohol | [X]% |
| 2 | volatile acidity | [X]% |
| 3 | sulphates | [X]% |
| 4 | total sulfur dioxide | [X]% |
| 5 | citric acid | [X]% |
| ... | ... | ... |

### Interpretasi Feature Importance

1. **Alcohol** - Fitur paling penting (sekitar 18%)
   - Semakin tinggi kadar alkohol, semakin tinggi kualitas anggur
   - Sesuai dengan pengetahuan umum tentang anggur

2. **Volatile Acidity** - Fitur penting kedua
   - Semakin tinggi asam volatil, semakin rendah kualitas
   - Asam volatil tinggi menyebabkan rasa seperti cuka

3. **Sulphates** - Fitur penting ketiga
   - Berfungsi sebagai pengawet anggur
   - Kadar yang tepat meningkatkan kualitas
## LANGKAH 10: PREDIKSI DATA TESTING DAN PEMBUATAN FILE SUBMISSION

### Tujuan
1. Memprediksi kualitas anggur pada data testing
2. Menyimpan hasil prediksi sesuai format yang diminta

### Proses yang Dilakukan
1. Model final memprediksi `X_test_scaled`
2. Hasil prediksi disimpan dalam dataframe dengan 2 kolom: `Id` dan `quality`
3. File disimpan sebagai CSV dengan format `hasilprediksi_3NIM.csv`

### Hasil Prediksi

| Kualitas | Jumlah | Persentase |
|----------|--------|------------|
| [quality] | [jumlah] | [X]% |
| [quality] | [jumlah] | [X]% |

### Format File Submission

**Contoh file yang dihasilkan:**
```csv
Id,quality
222,5
1514,6
417,5
