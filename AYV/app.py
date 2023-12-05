from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Load your machine learning model
model_path = r'AYV\Potato_leaf_prediction_model.h5'
model = tf.keras.models.load_model(model_path)

# Class mapping
class_mapping = {0: 'Potato___Early_Blight', 1: 'Potato___Late_Blight', 2: 'Potato___Healthy'}

# Ensure the 'uploads' directory exists
UPLOADS_FOLDER = 'uploads'
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        # Save the file
        file_path = os.path.join(UPLOADS_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess the image for your model
        img = image.load_img(file_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        predictions = model.predict(img_array)

        # Get the class with the highest probability
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Get the human-readable label
        predicted_class = class_mapping.get(predicted_class_index, 'Unknown')

        return render_template('index.html', message='Prediction: {}'.format(predicted_class))

if __name__ == '__main__':
    app.run(debug=True)
