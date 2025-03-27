import streamlit as st
import pickle
from pymongo import MongoClient

# Load trained vectorizer and model
with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)  # This ensures we use the same feature set

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# MongoDB connection
client = MongoClient("mongodb+srv://albeezsyedabdallah:Albeez_2001@fakenewscluster.tw442cb.mongodb.net/?retryWrites=true&w=majority&appName=FakeNewsCluster")
db = client["news_db"]
collection = db["news_results"]

# Function to scrape news (ensure it returns valid text)
def scrape_news(url):
    # Implement actual scraping logic here
    return "Sample scraped text from the URL"  # Replace with your real scraper

# Streamlit UI
st.title("Fake News Detector")

url = st.text_input("Enter News URL:")
user_text = st.text_area("Or enter the news text:")

if st.button("Analyze"):
    text = user_text if user_text else scrape_news(url)

    if not text.strip():
        st.error("No text found. Please enter valid text or URL.")
        st.stop()

    transformed_text = vectorizer.transform([text])  # Use the trained vectorizer

    if transformed_text.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch: expected {model.n_features_in_}, but got {transformed_text.shape[1]}")
        st.stop()

    prediction = model.predict(transformed_text)[0]
    result = "Yes" if prediction == 1 else "No"

    st.write(f"Fake News: {result}")
    collection.insert_one({"text": text, "result": result})

