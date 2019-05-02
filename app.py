# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 03:26:03 2019

@author: Utkarsh Kumar
"""

from flask import Flask,render_template,request
from sklearn.externals import joblib
import numpy as np



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html',template_folder='templates')

@app.route('/predict',methods=['POST'])
def predict():
    p_model = open('model.ml','rb')
    model = joblib.load(p_model)
    
    if request.method == 'POST':
        tv = request.form['tv']
        radio = request.form['radio']
        newspaper = request.form['newspaper']
        data= [tv,radio,newspaper]
        data = np.array(data)
        data = data.astype(np.float).reshape(1,-1)
        predict = model.predict(data)
    return render_template('index.html',template_folder='templates',prediction = predict)

if __name__ == '__main__':
    app.run(debug=False)

