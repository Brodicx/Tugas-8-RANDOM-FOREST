import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class ImageClassifier:
    def __init__(self):
        self.model = None
        self.classes = None
        
    def preprocess_image(self, image_path):
        """
        Preprocess gambar untuk model Random Forest
        """
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Tidak dapat membaca gambar")
            
        # Resize gambar ke ukuran yang konsisten
        img = cv2.resize(img, (64, 64))
        
        # Konversi ke grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Flatten gambar menjadi 1D array
        features = img.flatten()
        
        return features
    
    def train(self, X, y):
        """
        Melatih model Random Forest
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.classes = self.model.classes_
        
    def predict(self, image_path):
        """
        Melakukan prediksi pada gambar baru
        """
        if self.model is None:
            raise ValueError("Model belum dilatih")
            
        # Preprocess gambar
        features = self.preprocess_image(image_path)
        
        # Lakukan prediksi
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        
        return {
            'class': prediction,
            'confidence': float(max(probabilities))
        }
    
    def save_model(self, path):
        """
        Menyimpan model ke file
        """
        if self.model is None:
            raise ValueError("Model belum dilatih")
            
        model_data = {
            'model': self.model,
            'classes': self.classes
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        """
        Memuat model dari file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File model tidak ditemukan: {path}")
            
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.classes = model_data['classes'] 