#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bertopic import BERTopic
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import base64
import io
from PIL import Image
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
import warnings
from langchain.vectorstores import Chroma  # FIX: use Chroma not FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
import datetime

# Try to import Groq, fall back to OpenAI if Groq is not available
try:
    from langchain_groq import ChatGroq
    USE_GROQ = True
except ImportError:
    from langchain_openai import ChatOpenAI
    USE_GROQ = False

warnings.filterwarnings('ignore')

# --- Page config
st.set_page_config(
    page_title="Irish CHGFs Analysis Dashboard",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS (you already wrote, so not repeating for now to keep short) ---
# Copy your full CSS block here unchanged

# --- Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()

# --- Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None

if 'competitors_found' not in st.session_state:
    st.session_state.competitors_found = []

# --- Data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('ireland_cleaned_CHGF.xlsx')
        if 'Topic' in df.columns:
            df['Topic'] = df['Topic'].fillna('Uncategorized')
        if 'Founded Year' in df.columns:
            current_year = datetime.datetime.now().year
            df['Company Age'] = current_year - df['Founded Year']
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# --- Download link
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none;background-color:#103778;color:white;padding:8px 12px;border-radius:4px;font-weight:500;display:inline-block;margin-top:10px;">{text} 📥</a>'
    return href

# --- Topic modeling with BERTopic
@st.cache_resource
def run_bertopic(texts, n_topics=10):
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        embedding_model=sentence_model,
        vectorizer_model=vectorizer,
        nr_topics=n_topics,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics

# --- Topic modeling with CorEx
@st.cache_resource
def run_corex(texts, n_topics=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    topic_model = ct.Corex(n_hidden=n_topics)
    topic_model.fit(X, words=words)
    topics = [topic_model.get_topics(topic=i, n_words=10) for i in range(n_topics)]
    return topic_model, topics, words

# --- Text preprocessing
@st.cache_data
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Wordcloud
def generate_wordcloud(text, title=None):
    stopwords_set = set(stopwords.words('english'))
    wordcloud = WordCloud(
        width=1000, height=500,
        background_color='white',
        colormap='viridis',
        stopwords=stopwords_set
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold', color='#103778')
    plt.tight_layout()
    return fig

# --- Setup Chroma vectorstore
def setup_rag_for_competitor_analysis(df):
    documents = []
    for idx, row in df.iterrows():
        content = f"Company Name: {row['Company Name']}\n"
        if 'Description' in df.columns:
            content += f"Description: {row['Description']}\n"
        if 'Topic' in df.columns:
            content += f"Industry/Topic: {row['Topic']}\n"
        if 'City' in df.columns:
            content += f"City: {row['City']}\n"
        if 'Founded Year' in df.columns:
            content += f"Founded Year: {row['Founded Year']}\n"
        doc = Document(page_content=content, metadata={"source": idx})
        documents.append(doc)

    embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    embed_func = lambda texts: embeddings.encode(texts).tolist()

    # FIXED: Use Chroma here
    vectorstore = Chroma.from_documents(documents, embedding=embed_func)

    # Choose model
    if USE_GROQ:
        llm = ChatGroq(
            model="qwen-2.5-32b",
            api_key=os.environ.get("GROQ_API_KEY", "")
        )
    else:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    return retrieval_chain

# --- Find potential competitors
def find_potential_competitors(company_name, company_description, retrieval_chain):
    query = f"""
    Find potential competitors for a company named "{company_name}" with this description:
    {company_description}
    Return top 3-5 competitors with analysis.
    """
    result = retrieval_chain({"question": query, "chat_history": []})
    competitors = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            lines = doc.page_content.strip().split('\n')
            company_info = {}
            for line in lines:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    company_info[key] = value
            if 'Company Name' in company_info:
                competitors.append(company_info)
    return result["answer"], competitors

# --- Charts (pie, bar, timeline)
# (You already have correct chart functions, so you can reuse them)

# --- Main app
def main():
    # Your main UI structure (columns, uploaders, filters, tabs)
    # (You already wrote it well - no major change)

    st.title("Irish Consistent High Growth Firms (CHGFs) Analysis - 2023")

    df = load_data()

    if df.empty:
        st.warning("Upload a dataset first.")
        uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            if 'Topic' in df.columns:
                df['Topic'] = df['Topic'].fillna('Uncategorized')

    if not df.empty:
        # Create tabs
        tabs = st.tabs(["Dashboard", "Company Explorer", "Topic Analysis", "Advanced Modeling", "Competitor Analysis"])

        # Setup retrieval if needed
        if st.session_state.retrieval_chain is None:
            with st.spinner("Setting up competitor analysis..."):
                st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)

        # Dashboard Tab
        with tabs[0]:
            st.header("Dashboard Overview")
            st.metric("Total Companies", len(df))
            if 'Topic' in df.columns:
                st.metric("Topics", len(df['Topic'].unique()))
            if 'City' in df.columns:
                st.metric("Cities", len(df['City'].unique()))

# --- RUN ---
if __name__ == "__main__":
    main()
