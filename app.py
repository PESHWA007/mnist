
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:42:19 2022

@author: nishu
"""
from flask import Flask, render_template, request
from PIL import Image
import numpy as np

from keras.models import load_model
import tensorflow as tf


app = Flask(__name__, template_folder='templates')

def init():
    global model,graph
    model = load_model('model/model2.h5')
    graph = tf.compat.v1.get_default_graph()

@app.route('/')
def upload_file():
   return render_template('index.html')
	
@app.route('/sucess', methods = ['GET' ,'POST'])
def upload_image_file():
   if request.method == 'POST':
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        with graph.as_default():
            model = load_model('model/model2.h5')
            predict_x=model.predict(im2arr) 
            classes_x=np.argmax(predict_x,axis=1)

            return  render_template('sucess.html', classes_x=classes_x)
		
if __name__ == '__main__':
    print(("* Loading "))
    init()
    app.run(debug = True)


