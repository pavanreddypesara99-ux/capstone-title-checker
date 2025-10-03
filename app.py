import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Load data from CSV
@st.cache_data
def load_data(csv_url):
    df = pd.read_csv(csv_url)
    if "title" not in df.columns:
        raise ValueError("CSV must have a column named 'title'")
    return df

# Embed titles
@st.cache_resource
def embed_titles(titles, _model):
    return _model.encode(titles, convert_to_tensor=False)

# App layout
st.set_page_config(page_title="IBU Capstone Title Checker", page_icon="🧠")
st.title("🧠 IBU Capstone Title Similarity Checker")

csv_url = st.text_input("Dataset CSV URL", placeholder="Paste your published Google Sheet CSV link here")

if csv_url:
    try:
        df = load_data(csv_url)
        st.success(f"Loaded {len(df)} existing titles.")

        existing_titles = df["title"].astype(str).tolist()
        existing_embeddings = embed_titles(existing_titles, model)

        # User input
        query = st.text_input("Enter a proposed capstone title to check similarity")
        top_n = st.slider("How many similar titles to show", 1, 10, 5)
        threshold = st.slider("Flag if similarity ≥", 0.0, 1.0, 0.6)

        if query:
            query_embedding = embed_titles([query], model)
            similarities = cosine_similarity([query_embedding[0]], existing_embeddings)[0]

            results = pd.DataFrame({
                "Existing Title": existing_titles,
                "Similarity (%)": np.round(similarities * 100, 2)
            }).sort_values(by="Similarity (%)", ascending=False).reset_index(drop=True)

            # Highlight Top Match
            top_match = results.iloc[0]
            if top_match["Similarity (%)"] >= threshold * 100:
                st.error(f"⚠️ Too close! Your title is **{top_match['Similarity (%)']}% similar** to: *{top_match['Existing Title']}*")
            else:
                st.success(f"✅ Looks unique! Closest match is {top_match['Similarity (%)']}% similar: *{top_match['Existing Title']}*")

            # Show table of top N matches
            st.subheader("Top Similar Titles")
            st.dataframe(results.head(top_n))

    except Exception as e:
        st.error(f"Could not load data. Error: {e}")
