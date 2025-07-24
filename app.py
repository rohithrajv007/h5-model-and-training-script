import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# Path to the model file (now in root directory)
MODEL_PATH = "my_ECG_CNN.h5"
TARGET_SIZE = (128, 128)

# Class names for prediction
CLASS_NAMES = [
    'Left Bundle Branch Block',
    'Normal',
    'Premature Atrial Contraction',
    'Premature Ventricular Contractions',
    'Right Bundle Branch Block',
    'Ventricular Fibrillation'
]

# Load the model
model = None
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def preprocess_image(img):
    """Convert image to RGB, resize, normalize, and prepare for prediction."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Flask server is running. Use the /predict endpoint to classify ECG images.'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Upload .png, .jpg, or .jpeg only.'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        return jsonify({
            'predicted_class': predicted_class_name,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
