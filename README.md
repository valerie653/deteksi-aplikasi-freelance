# Deteksi Penipuan Lowongan Kerja Freelance

Aplikasi web berbasis Flask yang menggunakan Machine Learning (Logistic Regression + TF-IDF) untuk mendeteksi indikasi penipuan pada lowongan kerja freelance. Dibuat sebagai Penulisan Ilmiah — Universitas Gunadarma.

## Fitur

- Deteksi teks lowongan kerja secara real-time (asli / penipuan)
- Skor probabilitas, bukan sekadar label ya/tidak
- Deteksi kata kunci mencurigakan (red flags) dan tautan mencurigakan
- Analisis hybrid: TF-IDF + Logistic Regression + aturan red flags
- Endpoint `/health` untuk keperluan health check hosting

## Teknologi

- Python 3.11
- Flask 3.0
- scikit-learn (TF-IDF Vectorizer, Logistic Regression)
- Gunicorn (production server)
- HTML, CSS, JavaScript (vanilla, tanpa framework frontend)

## Struktur Proyek

```
.
├── app.py                     # Backend Flask (routes & fungsi prediksi)
├── requirements.txt           # Daftar dependensi Python
├── Procfile                   # Perintah menjalankan aplikasi di hosting
├── model_logreg.pkl           # Model Logistic Regression terlatih
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer terlatih
├── scaler.pkl                 # Scaler untuk fitur tambahan
└── templates/
    ├── index.html              # Halaman Beranda, Deteksi & Hasil Analisis
    └── about.html               # Halaman Tentang
```

## Menjalankan secara lokal

1. Clone atau download repository ini.
2. (Opsional tapi disarankan) buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   python app.py
   ```
5. Buka browser ke `http://localhost:5000`.

## Deploy ke Render.com (gratis)

1. Push repository ini ke GitHub.
2. Buat akun di [render.com](https://render.com), login dengan GitHub.
3. Klik **New +** → **Web Service**, hubungkan ke repository ini.
4. Isi konfigurasi:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT` (sudah ada di `Procfile`)
   - **Instance Type**: Free
5. Klik **Create Web Service** dan tunggu proses build selesai.
6. Aplikasi akan tersedia di `https://<nama-aplikasi>.onrender.com`.

Catatan: instance gratis Render akan "tidur" setelah idle beberapa waktu, sehingga request pertama setelah idle bisa terasa lambat (~30–60 detik). Ini normal untuk paket gratis.

## Endpoint API

| Method | Endpoint  | Deskripsi                                      |
|--------|-----------|-------------------------------------------------|
| GET    | `/`       | Halaman utama (Beranda, Deteksi, Hasil Analisis) |
| GET    | `/about`  | Halaman Tentang aplikasi                        |
| POST   | `/predict`| Menerima `{"text": "..."}`, mengembalikan hasil klasifikasi JSON |
| GET    | `/health` | Health check, mengembalikan `{"status": "ok"}`  |

Contoh request ke `/predict`:
```bash
curl -X POST https://<nama-aplikasi>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Dibutuhkan admin online, gaji besar tanpa pengalaman, langsung cair"}'
```

Contoh response:
```json
{
  "label": "PENIPUAN",
  "proba_fraud": 85.2,
  "proba_asli": 14.8,
  "threshold": 0.7,
  "rf_count": 4,
  "flags": ["tanpa pengalaman", "gaji besar", "langsung cair"],
  "links": []
}
```

## Model

Model dilatih menggunakan dataset [Real or Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) dari Kaggle (17.880 data). Proses training selengkapnya ada di `deteksi_lowongan_palsu.ipynb` (tidak disertakan di package deployment karena hanya dibutuhkan sekali saat training).

Hasil evaluasi model pada data uji (threshold optimal 0,70):

| Metrik    | Nilai  |
|-----------|--------|
| Akurasi   | 98,57% |
| Presisi   | 88,61% |
| Recall    | 80,92% |
| F1-Score  | 84,59% |

## Batasan

Aplikasi ini hanya menganalisis teks yang dimasukkan pengguna dan tidak melakukan pengecekan legalitas perusahaan secara langsung ke instansi terkait. Hasil analisis bersifat deteksi dini (early detection), bukan penentu akhir keabsahan suatu lowongan kerja.

## Lisensi

Dibuat untuk keperluan akademik (Penulisan Ilmiah), bebas digunakan untuk pembelajaran.
