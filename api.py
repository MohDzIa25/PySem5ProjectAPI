from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import io
from for_test_V20 import for_test

app = Flask(__name__)

def preprocess_image(img):
    img = img.convert('L')
    img_array = np.array(img)
    img_tensor = torch.from_numpy(img_array).type(torch.FloatTensor)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict_expression(img_tensor):
    # Assuming for_test is defined and imported correctly
    try:
        attention, prediction = for_test(img_tensor)
        prediction_string = 'Our Prediction: ' + prediction

        img_array = np.array(img_tensor.squeeze(0).squeeze(0))
        for i in range(attention.shape[0]):
            if i >= len(prediction):
                break
            elif prediction[i] == '<eol>':
                continue
            else:
                prediction_string += prediction[i]
        
    except Exception as e:
        # Handle the exception, you can print an error message or log it.
        # In this example, I'm just printing the exception message.
        # print(f"An error occurred ABC: {str(e)}")
        prediction_string = str(e)

    return prediction_string

@app.route('/', methods=['POST'])
def predict():
    try:
        # Check if the 'image' file is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        # Get the image file from the POST request
        img_file = request.files['image']

        # Read the image file as binary data
        image_data = img_file.read()

        # Open the image using PIL
        img = Image.open(io.BytesIO(image_data))

        # Preprocess the image (Replace preprocess_image with your actual preprocessing logic)
        img_tensor = preprocess_image(img)
        # print(img_tensor)

        # Make prediction (Replace predict_expression with your actual prediction logic)
        prediction_string = predict_expression(img_tensor)

        return jsonify({'prediction': prediction_string})

    except Exception as e:
        return jsonify({'error': str(e)})
        

if __name__ == '__main__':
    app.run(port=5069)
