from flask import Flask, request, jsonify, make_response
import tensorflow as tf
from tensorflow.keras import mixed_precision
from PIL import Image
import numpy as np
import io
import os

# --- Configuration ---
# Set up mixed precision for model loading
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

IMAGE_SIZE = 224

DISEASE_CLASSES = [
    'Bacterial gill disease', 'Bacterial red disease', 'Bacterial tail rot', 
    'Fungal diseases Saprolegniasis', 'Healthy Fish', 'Parasitic diseases', 
    'Viral diseases White tail disease'
]
SPECIES_CLASSES = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 
    'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout'
]

# --- Model Loading ---
# We load models ONCE when the serverless function starts.
# This makes subsequent requests much faster.
try:
    # Get the absolute path to the 'api' directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    disease_model_path = os.path.join(current_dir, 'disease_classifier_model.h5')
    species_model_path = os.path.join(current_dir, 'species_classifier_model.h5')
    
    disease_model = tf.keras.models.load_model(disease_model_path, compile=False)
    species_model = tf.keras.models.load_model(species_model_path, compile=False)
    
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_load_error = str(e)


# --- Preprocessing Function ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    img_batch = np.expand_dims(img_array, axis=0) # Create a batch
    return img_batch

# --- Initialize Flask App ---
# Vercel looks for an 'app' variable
app = Flask(__name__)


# --- Define API Endpoints ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET'])
def catch_all(path):
    # Redirect root requests or unknown paths,
    # Vercel should serve index.html automatically,
    # but this is a good fallback.
    return jsonify({'status': 'API is running. POST to /api/predict for inference.'})


@app.route('/api/predict', methods=['POST'])
def predict():
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        return make_response('', 204, headers)

    # Standard response headers
    response_headers = {'Access-Control-Allow-Origin': '*'}

    if not models_loaded:
        return jsonify({'error': f'Models failed to load: {model_load_error}'}), 500, response_headers

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400, response_headers
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400, response_headers

    try:
        image_bytes = file.read()
        tensor = preprocess_image(image_bytes)
        
        # --- Run Predictions ---
        disease_pred = disease_model.predict(tensor)[0]
        species_pred = species_model.predict(tensor)[0]
        
        # Get top prediction for disease
        disease_index = np.argmax(disease_pred)
        disease_class = DISEASE_CLASSES[disease_index]
        disease_conf = f"{disease_pred[disease_index]:.2%}"
        
        # Get top prediction for species
        species_index = np.argmax(species_pred)
        species_class = SPECIES_CLASSES[species_index]
        species_conf = f"{species_pred[species_index]:.2%}"
        
        # Return JSON response
        return jsonify({
            'disease': disease_class,
            'disease_confidence': disease_conf,
            'species': species_class,
            'species_confidence': species_conf
        }), 200, response_headers

    except Exception as e:
        return jsonify({'error': str(e)}), 500, response_headers

# This 'if' block is not strictly needed by Vercel,
# but it's good practice.
if __name__ == "__main__":
    app.run(debug=True)
