import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Initialize Flask App
app = Flask(__name__)


CORS(app)


MODEL_PATH = r"C:\Users\rajro\Downloads\-RhythmAI-ECG-Condition-Detector-Deep-Learning-Application-main\model\my_ECG_CNN.h5" 
TARGET_SIZE = (128, 128)
CLASS_NAMES = [
    'Left Bundle Branch Block',
    'Normal',
    'Premature Atrial Contraction',
    'Premature Ventricular Contractions',
    'Right Bundle Branch Block',
    'Ventricular Fibrillation'
]


model = None
try:
    
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    
    model = None


def preprocess_image(img):
    """
    Takes a PIL image, converts it to RGB, resizes it, and normalizes it
    for model prediction.
    """
   
    if img.mode != "RGB":
        img = img.convert("RGB")
    
   
    img = img.resize(TARGET_SIZE)
    
   
    img_array = image.img_to_array(img)
    
    
    img_array = np.expand_dims(img_array, axis=0)
    
    
    img_array /= 255.0
    
    return img_array



@app.route('/', methods=['GET'])
def index():
    """A simple root endpoint to check if the server is running."""
    return jsonify({'message': 'Flask server is running. Use the /predict endpoint to classify ECG images.'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image file in a POST request, preprocesses it,
    and returns the model's prediction as JSON.
    """
    
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

   
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload a .png, .jpg, or .jpeg image.'}), 400

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
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
