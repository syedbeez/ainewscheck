
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
from newspaper import Article
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Connect to MongoDB
client = MongoClient("mongodb+srv://albeezsyedabdallah:Albeez_2001@fakenewscluster.tw442cb.mongodb.net/?retryWrites=true&w=majority&appName=FakeNewsCluster")
db = client["news_db"]
collection = db["news_results"]

# Streamlit UI
st.title("AI-Powered News Verifier 📰")

# User input: URL or text
input_type = st.radio("Choose input type:", ("Enter Text", "Enter URL"))

if input_type == "Enter URL":
    url = st.text_input("Paste the article URL:")
    text = ""
    if st.button("Fetch & Analyze"):
        if url:
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
                st.write("Extracted Text:", text[:500] + "...")
            except Exception as e:
                st.error(f"Failed to extract article: {e}")
else:
    text = st.text_area("Enter news text:")

if text:
    # Preprocess input
    transformed_text = vectorizer.transform([text])

    # Predict
    result = model.predict(transformed_text)
    confidence = model.predict_proba(transformed_text).max()

    # Store in database
    collection.insert_one({"text": text, "result": int(result[0]), "confidence": confidence})

    # Display result
    label = "Real" if result[0] == 1 else "Fake"
    st.subheader(f"Prediction: **{label}** (Confidence: {confidence:.2f})")

    # Explainability with SHAP
    explainer = shap.LinearExplainer(model, vectorizer, feature_perturbation="interventional")
    shap_values = explainer.shap_values(transformed_text)

    st.subheader("Explainability: Important Words in the Prediction")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, transformed_text, feature_names=vectorizer.get_feature_names_out(), show=False)
    st.pyplot(fig)
