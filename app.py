from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model = load_model('C:/Users/Munish/Playground/flask/isl_model.keras')

# Define the path to your dataset
dataset_dir = Path('C:/Users/Munish/Playground/flask/ISL_Dataset/ISL_Dataset')

# Automatically infer the classes from the directory structure
class_names = sorted([item.name for item in dataset_dir.glob('*') if item.is_dir()])

# Create a label encoder with the inferred class names
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# Function to preprocess the input frame for prediction
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))  # Resize to match the model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to predict the class of a frame
def predict_sign_language(frame):
    img_array = preprocess_frame(frame)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    # Capture the image from the request
    file = request.files['image'].read()
    np_arr = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Predict the sign language
    predicted_label = predict_sign_language(frame)
    
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
