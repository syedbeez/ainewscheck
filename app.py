
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import pickle
import streamlit as st
import numpy as np
from newspaper import Article
import lime.lime_text
from pymongo import MongoClient
import datetime

# Load the vectorizer and model
with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# MongoDB Connection
client = MongoClient("mongodb+srv://albeezsyedabdallah:Albeez_2001@fakenewscluster.tw442cb.mongodb.net/?retryWrites=true&w=majority&appName=FakeNewsCluster")
db = client["news_db"]
collection = db["news_results"]

st.title("Fake News Detector 📰")

# Choose input method
input_type = st.radio("Select Input Type:", ["Enter Text", "Enter URL"])

text = ""
if input_type == "Enter Text":
    text = st.text_area("Enter news text:")
elif input_type == "Enter URL":
    url = st.text_input("Enter article URL:")
    if st.button("Fetch Article") and url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            st.success("Article extracted!")
        except Exception as e:
            st.error(f"Error fetching article: {e}")

# Predict button
if text and st.button("Predict"):
    # Transform input text
    transformed_text = vectorizer.transform([text])

    # Predict
    prediction = model.predict(transformed_text)[0]
    confidence = model.predict_proba(transformed_text).max()

    # Labels
    label = "🛑 Fake News" if prediction == 1 else "✅ Real News"

    # Show prediction
    st.subheader("Prediction Result:")
    st.write(f"**{label}** (Confidence: {confidence:.2f})")

    # LIME Explanation
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Real", "Fake"])
    explanation = explainer.explain_instance(text, model.predict_proba, num_features=5)
    
    st.subheader("Explainability (LIME):")
    st.write(explanation.as_list())
    st.pyplot(explanation.as_pyplot_figure())

    # Save to MongoDB
    doc = {
        "input": text[:500],  # Store first 500 characters
        "prediction": label,
        "confidence": confidence,
        "explanation": explanation.as_list(),
        "timestamp": datetime.datetime.utcnow()
    }
    collection.insert_one(doc)
    st.success("Saved to MongoDB!")
