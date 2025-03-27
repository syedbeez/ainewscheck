import streamlit as st
import shap
import torch
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix "no running event loop" issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Connect to MongoDB
client = MongoClient("mongodb+srv://albeezsyedabdallah:Albeez_2001@fakenewscluster.tw442cb.mongodb.net/?retryWrites=true&w=majority&appName=FakeNewsCluster")
db = client["news_db"]
collection = db["news_results"]

# Load your trained model & vectorizer
vectorizer = TfidfVectorizer()
model = LogisticRegression()

# Load actual data (replace with your dataset loading process)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("AI News Check - Fake News Detection")

# User input
text = st.text_area("Enter news text:", "")

if st.button("Check Authenticity"):
    transformed_text = vectorizer.transform([text])

    # Fix shape mismatch for SHAP explainability
    expected_shape = model.coef_.shape[1]  # Model expects same number of features
    input_shape = transformed_text.shape[1]

    if input_shape < expected_shape:
        padding = np.zeros((1, expected_shape - input_shape))  # Add zero padding
        transformed_text = np.hstack((transformed_text.toarray(), padding))
    else:
        transformed_text = transformed_text[:, :expected_shape]  # Trim excess features

    result = model.predict(transformed_text)
    probability = model.predict_proba(transformed_text)[0][1]

    st.subheader("Prediction Result:")
    st.write("✅ **Real News**" if result[0] == 1 else "❌ **Fake News**")
    st.write(f"Confidence Score: {probability:.2f}")

    # Save result to MongoDB
    collection.insert_one({"text": text, "result": int(result[0]), "confidence": float(probability)})

    # Explainability with SHAP
    explainer = shap.Explainer(model, vectorizer.transform(["sample text"]))  # Use sample input
    shap_values = explainer(transformed_text)

    st.subheader("Explainability: Important Words in the Prediction")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, transformed_text, feature_names=vectorizer.get_feature_names_out(), show=False)
    st.pyplot(fig)
