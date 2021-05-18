from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
model=pickle.load((open('model.pkl','rb')))
vect= pickle.load((open('vect.pkl','rb')))
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['Review']
    tx=[text]
    prediction= model.predict(vect.transform(tx))
    g=prediction[0]
    if g==1:
        Sentiment='Positive'
    else:
        Sentiment='Negative'
    return {
         "ACTUALL SENTENCE": text,
         "PREDICTED SENTIMENT": Sentiment,
    }

if __name__=="__main__":
    app.run()