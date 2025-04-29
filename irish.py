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
import re
import warnings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import datetime
import asyncio

# --- Streamlit Page Config (MUST BE FIRST ST COMMAND) ---
st.set_page_config(
    page_title="Irish CHGFs Analysis Dashboard",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- End Page Config ---


# --- NLTK Setup ---
@st.cache_resource
def download_nltk_resources():
    # ... (keep the function definition as before)
    import os
    import nltk
    needed = {'tokenizers/punkt': 'punkt', 'corpora/stopwords': 'stopwords'}
    downloaded_any = False
    # Use print for download status in the function, avoid st commands here
    print("Checking NLTK resources...")
    for resource_path, download_name in needed.items():
        try:
            nltk.data.find(resource_path)
            print(f"NLTK resource '{download_name}' found.")
        except LookupError:
            print(f"Downloading NLTK resource '{download_name}'...")
            try:
                nltk.download(download_name, quiet=True) # Use quiet=True for less terminal noise
                print(f"Successfully downloaded '{download_name}'.")
                downloaded_any = True
            except Exception as e:
                # Display error later in the main app body if needed
                print(f"ERROR: Failed to download NLTK resource '{download_name}': {str(e)}")
    return downloaded_any # Return status if needed


# Call the download function (uses print, not st)
NLTK_DOWNLOAD_ATTEMPTED = download_nltk_resources()

def safe_tokenize(text):
    # ... (keep the function definition as before)
    if text is None or not isinstance(text, str): return []
    try:
        nltk.data.find('tokenizers/punkt')
        return nltk.word_tokenize(text)
    except LookupError: # Fallback if still missing after initial download
        print("WARN: NLTK 'punkt' still not found during tokenization.")
        return re.findall(r'\b\w+\b', text.lower())
    except Exception as e:
        print(f"NLTK tokenization failed: {str(e)}. Using simple fallback.")
        return re.findall(r'\b\w+\b', text.lower())

def safe_get_stopwords():
    # ... (keep the function definition as before)
    try:
        nltk.data.find('corpora/stopwords')
        return set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        print("WARN: NLTK 'stopwords' still not found.")
        return set() # Return empty set if missing
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return set()

# --- End NLTK Setup ---


# --- LLM and Environment Variable Setup ---
# Check keys without using st commands yet
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_GROQ = bool(GROQ_API_KEY) # True if key exists
USE_OPENAI = bool(OPENAI_API_KEY) # True if key exists
RAG_ENABLED = USE_GROQ or USE_OPENAI # RAG possible if at least one key exists

# Import LLM classes (safe to do here)
llm_classes_imported = True
try:
    if USE_GROQ: from langchain_groq import ChatGroq
    if USE_OPENAI: from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"ERROR: Failed to import Langchain LLM classes: {e}")
    llm_classes_imported = False
    RAG_ENABLED = False # Disable RAG if imports fail

warnings.filterwarnings('ignore')
# --- End LLM Setup ---


# --- CSS Styling (st.markdown is fine after set_page_config) ---
st.markdown("""
<style>
    /* ... (keep all your CSS rules here) ... */
</style>
""", unsafe_allow_html=True)
# --- End CSS ---


# --- Display Setup Status (Now safe to use st commands) ---
# NLTK Verification
try:
    nltk.data.find('tokenizers/punkt')
    st.sidebar.success("NLTK Punkt Ready")
except LookupError:
    st.sidebar.error("NLTK Punkt Failed")
try:
    nltk.data.find('corpora/stopwords')
    st.sidebar.success("NLTK Stopwords Ready")
except LookupError:
    st.sidebar.error("NLTK Stopwords Failed")

# API Key / RAG Status
if not RAG_ENABLED:
    if llm_classes_imported:
        st.sidebar.error("No API Key (Groq/OpenAI). RAG Disabled.")
    else:
        st.sidebar.error("Langchain LLM Import Failed. RAG Disabled.")
else:
    # Indicate which keys are found
    key_status = []
    if USE_GROQ: key_status.append("Groq Key Found")
    if USE_OPENAI: key_status.append("OpenAI Key Found")
    st.sidebar.info("✔ " + " / ".join(key_status))


# --- Session State Initialization ---
# ... (keep this section as before) ...
if 'retrieval_chain' not in st.session_state: st.session_state.retrieval_chain = None
# ... etc ...


# --- Function Definitions ---
# ... (Keep all your function definitions: load_data, get_download_link, preprocess_text, etc. exactly as they were in the last full code block) ...
@st.cache_data
def load_data(uploaded_file=None):
    # ... function code ...
    pass
def get_download_link(df, filename, text):
     # ... function code ...
     pass
# etc... keep all functions


# --- Main App Logic ---
def main():
    # --- Page Title and Intro (Now safe) ---
    col1_title, col2_title = st.columns([1, 8], gap="small")
    # ... (keep title and flag code) ...

    # --- Data Input Section (Sidebar - st commands are fine now) ---
    st.sidebar.markdown("## 📂 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    df = load_data(uploaded_file) # Load and clean data

    # --- Stop if data loading failed ---
    if df.empty:
        st.error("No data loaded. Please upload a valid Excel file or ensure 'ireland_cleaned_CHGF.xlsx' exists.")
        st.stop()

    # --- Initialize RAG System (Now safe) ---
    if RAG_ENABLED and st.session_state.retrieval_chain is None:
        # Display status in sidebar now
        st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df) # setup_rag function now uses st.sidebar commands

    # --- Sidebar Filters (Now safe) ---
    st.sidebar.markdown("## 📊 Global Filters")
    # ... (keep all filter logic: multiselect, slider, text_input) ...
    df_filtered = df.copy()
    # ... apply filters ...
    st.sidebar.markdown(f"""...""") # Filtered count
    # ... download link ...


    # --- Main Content Tabs (Now safe) ---
    st.markdown("""...""") # Intro markdown
    tab_titles = ["📊 Dashboard", "🔍 Company Explorer", "🏷️ Topic Analysis", "🧠 Adv. Topic Modeling", "🥇 Competitor Analysis"]
    tabs = st.tabs(tab_titles)

    # ======== TAB 0: Dashboard ========
    with tabs[0]:
        # ... (Keep all the code for Tab 0 exactly as before) ...
        pass

    # ======== TAB 1: Company Explorer ========
    with tabs[1]:
        # ... (Keep all the code for Tab 1 exactly as before) ...
        pass

    # ======== TAB 2: Topic Analysis ========
    with tabs[2]:
        # ... (Keep all the code for Tab 2 exactly as before, including the defensive initialization) ...
        pass

    # ======== TAB 3: Advanced Topic Modeling ========
    with tabs[3]:
        # ... (Keep all the code for Tab 3 exactly as before) ...
        pass

    # ======== TAB 4: Competitor Analysis ========
    with tabs[4]:
        # ... (Keep all the code for Tab 4 exactly as before) ...
        pass


# --- App Entry Point ---
if __name__ == "__main__":
    # ... (keep the asyncio event loop handling) ...
    main()