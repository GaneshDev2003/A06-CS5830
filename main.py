from typing import Union
import sys
from fastapi import FastAPI, UploadFile, File
import uvicorn
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

def main(args):
    app = FastAPI()
    
    # Load the model using a h5 file given by the path
    def load_module(path):
        model = load_model(path)
        return model
    
    # Format the image to a 28x28 grayscale image
    def format_image(image):
        image = image.resize((28, 28))
        image = image.convert('L')
        return image
    
    # Predict the digit given an array of shape (1, 784) and a keras model
    def predict_digit(image_array, model):
        probs = model.predict(image_array, verbose=True)
        print("Predicted Digit:", np.argmax(probs))
        return str(np.argmax(probs))
    
    # Endpoint to predict the digit given an image file (png, jpg, jpeg)
    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        # load model from h5 file given by the path
        model = load_module(args[1])
        contents = await file.read()
        # Convert image into a PIL image
        image = Image.open(io.BytesIO(contents))
        image = format_image(image)
        image_array = np.array(image)
        # Flatten the 28x28 image into a 1x784 array
        image_array = image_array.flatten().reshape(1,784)
        # Normalize the image array
        image_array = image_array/255.0
        return {"digit": predict_digit(image_array, model)}

    uvicorn.run(app)

if __name__ == "__main__":
    main(sys.argv)