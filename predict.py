import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model_path = "models/crop_disease_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = load_model(model_path)
print("✅ Model loaded successfully.")

# Class names
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Function to load and preprocess image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image. Make sure it's a valid image file.")

    print(f"📸 Loaded image: {image_path}")
    print(f"Image shape before resize: {img.shape}")

    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    print(f"Image shape after preprocess: {img.shape}")
    return img

# === Change this to your image file name ===
image_path = "predict6.jpg"

# Run prediction
try:
    image = preprocess_image(image_path)
    prediction = model.predict(image)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])

    print("\n🎯 Prediction Result:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw prediction vector: {prediction}")

except Exception as e:
    print(f"❌ Error: {e}")
