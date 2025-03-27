import pickle
import streamlit as st
import numpy as np
from newspaper import Article
import lime.lime_text
from pymongo import MongoClient
import datetime
from scipy.sparse import hstack, csr_matrix

# Load vectorizer and model
with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# MongoDB Connection
client = MongoClient("mongodb+srv://albeezsyedabdallah:Albeez_2001@fakenewscluster.tw442cb.mongodb.net/?retryWrites=true&w=majority&appName=FakeNewsCluster")
db = client["news_db"]
collection = db["news_results"]

st.title("📰 Fake News Detector")

# Choose input type
input_type = st.radio("Select Input Type:", ["Enter Text", "Enter URL"])

# Initialize session state for text
if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""

if input_type == "Enter Text":
    text = st.text_area("Enter news text:", value=st.session_state["article_text"])
else:
    url = st.text_input("Enter article URL:")
    if st.button("Fetch Article") and url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            st.session_state["article_text"] = article.text
            st.success("Article extracted! Now click 'Predict'.")
        except Exception as e:
            st.error(f"Error fetching article: {e}")

# Use session state text for prediction
text = st.session_state["article_text"]

# Predict button
if text and st.button("Predict"):
    if not text.strip():
        st.error("Please enter or fetch some text before predicting.")
    else:
        # Transform text
        transformed_text = vectorizer.transform([text])  
        
        # Get feature counts
        model_features = model.n_features_in_
        vectorizer_features = transformed_text.shape[1]

        # Dynamically match features
        if vectorizer_features < model_features:
            # Pad with zeros if features are less
            padding = csr_matrix((1, model_features - vectorizer_features))
            transformed_text = hstack([transformed_text, padding])
        elif vectorizer_features > model_features:
            # Trim extra features if needed
            transformed_text = transformed_text[:, :model_features]

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
        explanation = explainer.explain_instance(text, lambda x: model.predict_proba(vectorizer.transform(x)))



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
