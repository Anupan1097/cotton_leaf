# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3

from flask import Flask, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

model_path = 'cotton_leaf_V3.h5'

model = load_model(model_path)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size = (224, 224))
    
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis = 0)
    
    preds = model.predict(x)
    preds = np.argmax(preds, axis = 1)
    
    if preds == 0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
        
        
    return preds

@app.route('/', methods = ['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(filepath)
        
        preds = model_predict(filepath, model)
        
        result = preds
        
        return result
    
    return None

if __name__ =="__main__":
    app.run()



    
    
        
        
        
        
        
        
        
        
        
    
    
    
