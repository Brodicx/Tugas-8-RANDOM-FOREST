from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from models.model import ImageClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inisialisasi classifier
classifier = ImageClassifier()
model_path = 'models/random_forest_model.joblib'

# Coba muat model jika ada
if os.path.exists(model_path):
    try:
        classifier.load_model(model_path)
        print("Model berhasil dimuat")
    except Exception as e:
        print(f"Error memuat model: {str(e)}")
else:
    print("Model belum dilatih. Silakan jalankan train_model.py terlebih dahulu.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluation')
def evaluation():
    cm_image_path = '/static/plots/confusion_matrix.png'
    # Pastikan file gambar ada sebelum menampilkannya
    if not os.path.exists('static/plots/confusion_matrix.png'):
        cm_image_path = None # Atau berikan placeholder/pesan error
    return render_template('evaluation.html', cm_image=cm_image_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Lakukan prediksi
            result = classifier.predict(filepath)
            
            return jsonify({
                'prediction': f"Kelas: {result['class']} (Confidence: {result['confidence']:.2f})",
                'image_path': f'/static/uploads/{filename}'
            })
        except Exception as e:
            return jsonify({'error': f'Error saat prediksi: {str(e)}'})
    
    return jsonify({'error': 'Format file tidak didukung'})

if __name__ == '__main__':
    app.run(debug=True) 