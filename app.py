from cmath import log
from json import load
import re
from flask import Flask, request , render_template , jsonify
import sys
import os
import pandas as pd
from sklearn import preprocessing
from IMDB.app_constants import ROOT_DIR 
from IMDB.app_util.util import load_object, save_object
from IMDB.app_exception.exception import App_Exception
from nltk.tokenize.toktok import ToktokTokenizer
import logging
from bs4 import BeautifulSoup
import re 
import nltk 
nltk.download('stopwords')


ROOT_DIR = ROOT_DIR = os.getcwd()
TRAINED_MODEL_dir = os.path.join(ROOT_DIR, 'trained_model')
TRAINED_MODEL_PATH = os.path.join(TRAINED_MODEL_dir, 'model.pkl')


app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            review = request.form['review']
            review_dict = {'review': review}
            X = pd.DataFrame(review_dict , index=[0])
            model =load_object(file_path=TRAINED_MODEL_PATH )
            transformed_feature = model.preprocessing_object.transform(X)
            predicted = model.trained_model_object.predict(transformed_feature)
            prediction = model.label_binarizer_object.inverse_transform(predicted).tolist()[0]
            return jsonify({"Prediction" : prediction})
        else:
            return render_template('index.html')
    except Exception as e:
        return str(e)




if __name__ == "__main__":
    app.run()
