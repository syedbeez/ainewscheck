import streamlit as st
import pickle
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from pymongo import MongoClient

# Load TF-IDF vectorizer and trained model
with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# MongoDB connection
client = MongoClient("mongodb+srv://albeezsyedabdallah:Albeez_2001@fakenewscluster.tw442cb.mongodb.net/?retryWrites=true&w=majority&appName=FakeNewsCluster")
db = client["news_db"]
collection = db["news_results"]

# Function to extract BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# Streamlit UI
st.title("Fake News Detector")

url = st.text_input("Enter News URL:")
user_text = st.text_area("Or enter the news text:")

if st.button("Analyze"):
    text = user_text.strip()
    
    if not text:
        st.error("Please enter valid news text.")
        st.stop()
    
    # Transform input using trained TF-IDF vectorizer
    transformed_text = vectorizer.transform([text]).toarray()

    # Extract BERT embeddings for the input
    bert_features = get_bert_embedding(text).reshape(1, -1)

    # Combine TF-IDF and BERT features
    combined_features = np.hstack([transformed_text, bert_features])

    # Ensure the feature size matches the model's training size
    if combined_features.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch: expected {model.n_features_in_}, but got {combined_features.shape[1]}")
        st.stop()

    # Make prediction
    prediction = model.predict(combined_features)[0]
    result = "Fake News" if prediction == 1 else "Real News"

    # Display result
    st.write(f"Prediction: {result}")
    collection.insert_one({"text": text, "result": result})
