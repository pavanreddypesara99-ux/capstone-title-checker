# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="IBU Capstone Title Checker", page_icon="🧠", layout="wide")
st.title("🧠 IBU Capstone Title Similarity Checker")

# --- 1) Where is your dataset? ---
# Easiest: keep your titles in a Google Sheet published as CSV.
# You can replace DEFAULT_CSV_URL with your own link (or type it in the sidebar).
DEFAULT_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQQAoO_eJz3idWJSu4PVCzgBgEw_NDFwFgNiAOAGoQSvkvTMdZyxwVHiHSuPseZEvpoH6Z8SKDF077b/pub?output=csv"
)

csv_url_from_secret = st.secrets.get("CSV_URL", "")
data_url = st.sidebar.text_input("Dataset CSV URL", value=csv_url_from_secret or DEFAULT_CSV_URL)

st.sidebar.markdown(
    "Tip: Put your Google Sheet in one column called **title**. "
    "Publish to web → CSV, then paste the link here."
)

# --- 2) Load model & data (cached so it’s fast) ---
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(ttl=600)
def load_titles(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Make sure we have a 'title' column
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=cols_lower, inplace=True)
    if "title" not in df.columns:
        # Assume first column is the title if not labeled
        df.rename(columns={df.columns[0]: "title"}, inplace=True)
    df["title"] = df["title"].astype(str).str.strip()
    df.dropna(subset=["title"], inplace=True)
    df.drop_duplicates(subset=["title"], inplace=True)
    return df[["title"]]

@st.cache_resource
def embed_titles(model, titles: list[np.ndarray]):
    return model.encode(titles, normalize_embeddings=True)

# Load
try:
    model = load_model()
    titles_df = load_titles(data_url)
    title_list = titles_df["title"].tolist()
    title_embeddings = embed_titles(model, title_list)
except Exception as e:
    st.error(f"Could not load data/model. Check your CSV URL. Details: {e}")
    st.stop()

st.success(f"Loaded {len(title_list)} existing titles.")

# --- 3) Query UI ---
query = st.text_input("Enter a **proposed capstone title** to check similarity")
top_k = st.slider("How many similar titles to show", 3, 25, 10)
threshold = st.slider("Flag if similarity ≥", 0.00, 1.00, 0.60, 0.01)

# --- 4) Compute similarity ---
if query:
    q_vec = model.encode([query], normalize_embeddings=True)
    # cosine similarities (dot product because both normalized)
    sims = (q_vec @ title_embeddings.T)[0]
    out = pd.DataFrame(
        {"Existing Title": title_list, "Similarity": sims}
    ).sort_values("Similarity", ascending=False)
    out["Similarity (%)]"] = (out["Similarity"] * 100).round(2)

    # Show results
    st.subheader("Closest matches")
    st.dataframe(out.head(top_k)[["Existing Title", "Similarity (%)]"]], use_container_width=True)

    # Simple verdict
    max_sim = float(out["Similarity"].max())
    if max_sim >= threshold:
        st.warning(f"⚠️ High similarity detected (max={max_sim:.2f}). Consider revising your title.")
    else:
        st.success(f"✅ Looks unique enough (max similarity={max_sim:.2f}).")
else:
    st.info("Type a proposed title above to see similar ones.")
