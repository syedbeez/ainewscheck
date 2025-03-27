import streamlit as st
import requests
import pickle
import shap
from bs4 import BeautifulSoup
from pymongo import MongoClient

#Load Model and Vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))
explainer = pickle.load(open("shap_explainer.pkl", "rb"))

st.title("Fake News Detector")
url = st.text_input("Enter News URL")
user_text = st.text_area("Or Enter News Text Directly")

client = MongoClient("mongodb+srv://albeezsyedabdallah:Albeez_2001@fakenewscluster.tw442cb.mongodb.net/?retryWrites=true&w=majority&appName=FakeNewsCluster")
db = client["FakeNewsDB"]
collection = db["NewsArticles"]

def scrape_news(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    return " ".join([h2.get_text(strip=True) for h2 in soup.find_all('h2')])
 
if st.button("Analyze"):
    text = user_text if user_text else scrape_news(url)
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    result = "Yes" if prediction == 1 else "No"
    st.write(f"Fake News: {result}")
    collection.insert_one({"text": text, "result": result})
    shap_values = explainer(transformed_text)
    shap.summary_plot(shap_values, transformed_text)
