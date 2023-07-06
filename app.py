import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from PIL import Image

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def preprocess_image(image):
    # Resize the image to the desired input shape
    image = image.resize((256, 256))
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Normalize the image array if needed
    # ...

    return image_array

def threshold_prediction(prediction, threshold=0.5):
    if prediction >= threshold:
        return 1
    else:
        return 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get the uploaded image file
    image_file = request.files['image']
    
    # Open the image using PIL
    image = Image.open(image_file)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Reshape the processed image to match the expected input shape
    reshaped_image = processed_image.reshape((1, 256, 256, 3))

    # Pass the reshaped image to the model for prediction
    prediction = model.predict(reshaped_image)

    # Perform any necessary post-processing on the prediction
    thresholded_prediction = threshold_prediction(prediction)

    # Return the prediction result to the HTML template
    return render_template('index.html', prediction_text='Prediction: {}'.format(thresholded_prediction))

if __name__ == "__main__":
    app.run(debug=True)
