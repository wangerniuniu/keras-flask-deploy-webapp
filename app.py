import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import torch
import torchvision
from torchvision import transforms
# import tensorflow as tf
# from tensorflow import keras

# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from torchvision.models import resnet50
# model = MobileNetV2(weights='imagenet')
model = resnet50(pretrained=True).cuda()
model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), normalize]
)

print('Model loaded. Check http://127.0.0.1:423/')


# Model saved with Keras model.save()
# MODEL_PATH = 'models/your_model.h5'

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


def model_predict(img, model):
    # 对图像进行归一化
    img_p = transform(img.convert('RGB'))
    print(img_p.shape)
    
    # 增加一个维度
    img_normalize = torch.unsqueeze(img_p, 0).cuda()
    preds = model(img_normalize)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        base_dir = os.path.dirname(__file__)
        # resp = make_response(open(os.path.join(base_dir, "imagenet_classes.txt")).read())

        with open(os.path.join(base_dir, "imagenet_classes.txt")) as f:
            classes = [line.strip() for line in f.readlines()]
        
        _, indices = torch.sort(preds, descending=True)
        percentage = torch.nn.functional.softmax(preds, dim=1)[0] * 100
        idx=indices[0][0]
        prediction = classes[idx]
        score = percentage[idx].item()
        print(prediction,score)
        return jsonify(result=prediction, probability=score)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 423), app)
    http_server.serve_forever()
