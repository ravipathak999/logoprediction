from fastapi import FastAPI, File, UploadFile
import shutil
import os
import cv2
import numpy as np
import tensorflow as tf

app = FastAPI()

# Suppress TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

model = tf.keras.models.load_model("/Users/pythonista/logo_detection_model.h5")

# Define the directory to store temporary files
TMP_DIR = "/tmp"

# Define the input size expected by the model
INPUT_SIZE = (100, 100)

# Define the classes
classes = ['Starbucks', 'Nike', 'Burger King', 'Dunkin\' Donuts', 'Pepsi', 'McDonald\'s', 'Coca-Cola']

def preprocess_image(image, size):
    # Resize the input image to the fixed size expected by the model
    resized_image = cv2.resize(image, size)
    # Normalize the pixel values
    normalized_image = resized_image / 255.0
    # Add batch dimension
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

def predict_logo(image):
    preprocessed_image = preprocess_image(image, INPUT_SIZE)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    return predicted_class

def detect_logo(file_path):
    try:
        # Read the uploaded image
        image = cv2.imread(file_path)
        # Detect the logo
        predicted_class = predict_logo(image)
        class_name = classes[predicted_class]
        return class_name
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary directory
        file_path = os.path.join(TMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect the logo in the uploaded file
        class_name = detect_logo(file_path)

        # Delete the temporary file
        os.remove(file_path)

        return {"message": f"Successfully uploaded {file.filename}. Predicted class: {class_name}"}
    except Exception as e:
        return {"message": f"There was an error processing the image: {str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
