import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)
from user import text_processing
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    features = [x for x in request.form.values()]
    final_features = {'title':features[1],'text':features[0]}
    text_processing(final_features)
    prediction = model.predict_one(final_features)

    #output = round(prediction[0], 2)

    return render_template('index.html',prediction_text='The News is {}'.format(prediction))


if __name__ == "__main__":
    #text_processing(dataset, Y=None)
    app.run(debug=True)
