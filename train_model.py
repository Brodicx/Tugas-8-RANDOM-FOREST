import os
import numpy as np
from models.model import ImageClassifier
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def load_dataset(dataset_path):
    """
    Memuat dataset dari folder
    Format folder:
    dataset_path/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
    """
    X = []
    y = []
    
    # Iterasi melalui setiap kelas
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Iterasi melalui setiap gambar dalam kelas
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                # Preprocess gambar
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Tidak dapat membaca gambar: {image_path}")
                    continue
                    
                img = cv2.resize(img, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                features = img.flatten()
                
                X.append(features)
                y.append(class_name)
                
            except Exception as e:
                print(f"Error memproses {image_path}: {str(e)}")
                continue
    
    return np.array(X), np.array(y)

def main():
    # Path ke dataset
    dataset_path = "dataset"  # Sesuaikan dengan path dataset Anda
    
    print("Memuat dataset...")
    X, y = load_dataset(dataset_path)
    
    if len(X) == 0:
        print("Tidak ada gambar yang berhasil dimuat!")
        return
        
    print(f"Dataset dimuat: {len(X)} gambar dari {len(np.unique(y))} kelas")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Inisialisasi dan latih model
    print("Melatih model...")
    classifier = ImageClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluasi model
    print("\nEvaluasi model:")
    train_accuracy = classifier.model.score(X_train, y_train)
    test_accuracy = classifier.model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
    
    # Visualisasi Confusion Matrix
    print("Membuat Confusion Matrix...")
    y_pred = classifier.model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes)
    
    # Cetak Laporan Klasifikasi
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=classifier.classes))
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Klasifikasi Gambar')
    
    plot_dir = 'static/plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    print(f"Confusion Matrix disimpan ke: {os.path.join(plot_dir, 'confusion_matrix.png')}")
    plt.close() # Tutup plot agar tidak muncul terus-menerus
    
    # Simpan model
    model_path = "models/random_forest_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)
    print(f"\nModel disimpan ke: {model_path}")

if __name__ == "__main__":
    main() 