import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# === BACA DATASET ===
df = pd.read_csv("dataset_game.csv")

# === EKSTRAKSI FITUR GAMBAR (warna rata-rata RGB) ===
fitur = []
label = []

for i, row in df.iterrows():
    file_path = os.path.join("data", row["file"])
    try:
        img = Image.open(file_path).resize((64, 64)).convert("RGB")
        arr = np.array(img)
        mean_rgb = arr.mean(axis=(0, 1))  # Rata-rata warna
        fitur.append(mean_rgb)
        label.append(row["jawaban"].upper())
    except Exception as e:
        print(f"Gagal memproses gambar: {row['file']} - {e}")

X = np.array(fitur)
y = np.array(label)

# === ENCODING LABEL ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === TRAIN MODEL ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === EVALUASI SINGKAT ===
accuracy = model.score(X_test, y_test)
print(f"✅ Akurasi model: {accuracy:.2f}")

# === SIMPAN MODEL & ENCODER ===
joblib.dump(model, "model_sayur.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model dan LabelEncoder berhasil disimpan.")
