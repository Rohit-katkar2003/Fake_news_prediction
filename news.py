import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

# Load the trained model
loaded_model = pickle.load(open("C:/Users/rohit/Data science/Major Project/.ipynb_checkpoints/Fake News Prediction/news.sav", 'rb'))

# Load the fitted vectorizer
vectorizer = pickle.load(open("C:/Users/rohit/Data science/Major Project/.ipynb_checkpoints/Fake News Prediction/vectorizer.sav", 'rb'))

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred_DT = loaded_model.predict(new_xv_test)
    return output_label(pred_DT[0])

st.title('Fake News Prediction')
inp_data = st.text_area('Enter News:')

if st.button('Predict'):
    if inp_data:
        prediction = manual_testing(inp_data)
        if prediction == "Fake News":
            st.error(f'Prediction: {prediction}')
        else:
            st.success(f'Prediction: {prediction}')
        
