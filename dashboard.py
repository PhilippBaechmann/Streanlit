#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import base64
import re
import warnings
import os
import datetime
import io
import nltk

from PIL import Image
from wordcloud import WordCloud
from bertopic import BERTopic
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain

try:
    from langchain_groq import ChatGroq
    USE_GROQ = True
except ImportError:
    from langchain_openai import ChatOpenAI
    USE_GROQ = False

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Irish CHGFs Analysis Dashboard",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()

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
    except:
        return pd.DataFrame()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'competitors_found' not in st.session_state:
    st.session_state.competitors_found = []

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_wordcloud(text, title=None):
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    wordcloud = WordCloud(
        width=1000, height=500, background_color='white',
        stopwords=stopwords_set, colormap='viridis'
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold', color='#103778')
    return fig

def run_bertopic(texts, n_topics=10):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer = CountVectorizer(stop_words="english")
    topic_model = BERTopic(embedding_model=model, vectorizer_model=vectorizer, nr_topics=n_topics)
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics

def run_corex(texts, n_topics=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    topic_model = ct.Corex(n_hidden=n_topics)
    topic_model.fit(X, words=words)
    topics = [topic_model.get_topics(topic=i, n_words=10) for i in range(n_topics)]
    return topic_model, topics

def create_charts(df):
    if 'City' in df.columns:
        city_counts = df['City'].value_counts()
        if len(city_counts) > 10:
            city_counts = city_counts.head(10).append(pd.Series(city_counts[10:].sum(), index=['Others']))
        fig1 = px.pie(values=city_counts.values, names=city_counts.index, title='Companies by City', hole=0.4)
        st.plotly_chart(fig1)

    if 'Founded Year' in df.columns:
        year_counts = df['Founded Year'].value_counts().sort_index()
        fig2 = px.line(x=year_counts.index, y=year_counts.values, markers=True, title='Foundations Over Time')
        st.plotly_chart(fig2)

def setup_rag(df):
    documents = []
    for idx, row in df.iterrows():
        content = f"Company Name: {row['Company Name']}\n"
        if 'Description' in df.columns:
            content += f"Description: {row['Description']}\n"
        if 'Topic' in df.columns:
            content += f"Topic: {row['Topic']}\n"
        if 'City' in df.columns:
            content += f"City: {row['City']}\n"
        if 'Founded Year' in df.columns:
            content += f"Founded Year: {row['Founded Year']}\n"
        doc = Document(page_content=content)
        documents.append(doc)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    if USE_GROQ:
        llm = ChatGroq(model="qwen-2.5-32b", api_key=os.getenv("GROQ_API_KEY"))
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), return_source_documents=True
    )
    return retrieval_chain

def main():
    df = load_data()

    if df.empty:
        st.warning("No data found. Upload your Excel file.")
        uploaded = st.file_uploader("Upload", type=["xlsx"])
        if uploaded:
            df = pd.read_excel(uploaded)

    if not df.empty:
        st.title("Irish CHGFs Dashboard")
        tabs = st.tabs(["Dashboard", "Company Explorer", "Topic Modeling", "Advanced Modeling", "Competitor Analysis"])

        if st.session_state.retrieval_chain is None:
            with st.spinner("Setting up competitor analysis..."):
                st.session_state.retrieval_chain = setup_rag(df)

        with tabs[0]:
            st.header("Overview")
            st.metric("Total Companies", len(df))
            create_charts(df)

        with tabs[1]:
            st.header("Company Explorer")
            st.dataframe(df)

        with tabs[2]:
            st.header("Topic Modeling (BERTopic)")
            texts = df['Description'].dropna().tolist() if 'Description' in df.columns else []
            if texts:
                topic_model, topics = run_bertopic(texts)
                st.write(topic_model.get_topic_info())

        with tabs[3]:
            st.header("Topic Modeling (CorEx)")
            texts = df['Description'].dropna().tolist() if 'Description' in df.columns else []
            if texts:
                topic_model, topics = run_corex(texts)
                st.write(topics)

        with tabs[4]:
            st.header("Competitor Analysis")
            company_name = st.text_input("Company Name")
            description = st.text_area("Description")
            if st.button("Find Competitors"):
                query = f"Find competitors for {company_name}. Description: {description}"
                result = st.session_state.retrieval_chain({"question": query, "chat_history": []})
                st.write(result['answer'])

if __name__ == "__main__":
    main()

