# Klasifikasi Gambar dengan Random Forest

Aplikasi web untuk klasifikasi gambar menggunakan algoritma Random Forest. Aplikasi ini dibangun dengan Flask dan menggunakan Scikit-Learn untuk implementasi Random Forest.

## Fitur

- Upload gambar melalui web interface
- Drag and drop support
- Preview gambar sebelum prediksi
- Klasifikasi gambar menggunakan Random Forest
- Menampilkan hasil prediksi dengan confidence score

## Persyaratan

- Python 3.8+
- Flask
- OpenCV
- Scikit-Learn
- Pandas
- NumPy
- Gunicorn (untuk production)

## Instalasi

1. Clone repository ini
2. Buat virtual environment (opsional tapi direkomendasikan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Persiapan Dataset

1. Buat folder `dataset` di root proyek
2. Di dalam folder `dataset`, buat subfolder untuk setiap kelas
3. Masukkan gambar-gambar ke dalam folder kelas yang sesuai
4. Format folder harus seperti ini:
   ```
   dataset/
       class1/
           image1.jpg
           image2.jpg
           ...
       class2/
           image1.jpg
           image2.jpg
           ...
   ```

## Melatih Model

Jalankan script training:
```bash
python train_model.py
```

Model yang telah dilatih akan disimpan di `models/random_forest_model.joblib`

## Menjalankan Aplikasi

### Development
```bash
python app.py
```

### Production
```bash
gunicorn app:app
```

Aplikasi akan berjalan di `http://localhost:5000`

## Struktur Proyek

```
.
├── models/
│   ├── __init__.py
│   ├── model.py
│   └── random_forest_model.joblib
├── static/
│   ├── css/
│   └── uploads/
├── templates/
│   └── index.html
├── dataset/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Deployment

Aplikasi ini dapat di-deploy ke berbagai platform cloud seperti:
- Heroku
- Google Cloud Platform
- AWS
- DigitalOcean

Pastikan untuk mengatur environment variables yang diperlukan dan menggunakan Gunicorn sebagai production server.

## Lisensi

MIT License 