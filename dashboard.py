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
import os
import datetime

from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

# Try to import Groq, fall back to OpenAI if Groq is not available
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
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'competitors_found' not in st.session_state:
    st.session_state.competitors_found = []

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
        other_cols = [col for col in df.columns if col not in ['Company Name', 'Description', 'Topic', 'City', 'Founded Year']]
        for col in other_cols:
            if not pd.isna(row[col]):
                content += f"{col}: {row[col]}\n"
        documents.append(Document(page_content=content, metadata={"source": idx}))

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    if USE_GROQ:
        llm = ChatGroq(model="qwen-2.5-32b", api_key=os.environ.get("GROQ_API_KEY", ""))
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY", ""))

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    return retrieval_chain

def main():
    df = load_data()

    if df.empty:
        st.warning("Please upload a valid Excel file containing the Irish CHGFs data.")
        uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)

    if not df.empty:
        tabs = st.tabs(["Dashboard", "Company Explorer", "Topic Analysis", "Advanced Modeling", "Competitor Analysis"])

        if st.session_state.retrieval_chain is None:
            with st.spinner("Setting up competitor analysis system..."):
                st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)

        with tabs[0]:
            st.header("Dashboard Overview")
            st.metric("Total Companies", len(df))

if __name__ == "__main__":
    main()
