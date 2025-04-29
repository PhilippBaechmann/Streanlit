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
# from langchain.text_splitter import CharacterTextSplitter # Not explicitly used, can be removed if not needed elsewhere
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import datetime
import asyncio # Added for potential event loop issues in some environments

# First define NLTK resource download function
@st.cache_resource
def download_nltk_resources():
    import os
    import nltk

    try:
        # Check if punkt exists, if not download it
        try:
            nltk.data.find('tokenizers/punkt')
            print("NLTK punkt tokenizer already downloaded")
        except LookupError:
            nltk.download('punkt', quiet=False)
            print("Downloaded NLTK punkt tokenizer")

        # Check if stopwords exists, if not download it
        try:
            nltk.data.find('corpora/stopwords')
            print("NLTK stopwords already downloaded")
        except LookupError:
            nltk.download('stopwords', quiet=False)
            print("Downloaded NLTK stopwords")
    except Exception as e:
        print(f"Failed to download NLTK resources: {str(e)}")

# Define safe tokenize function
def safe_tokenize(text):
    if text is None:
        print("Warning: Text is None")
        return []

    if not isinstance(text, str):
        print(f"Warning: Text is not a string, but {type(text)}")
        return []

    try:
        # Ensure resources are available before tokenizing
        nltk.data.find('tokenizers/punkt')
        return nltk.word_tokenize(text)
    except LookupError:
        print("NLTK punkt tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
        return nltk.word_tokenize(text) # Try again after download
    except Exception as e:
        print(f"NLTK tokenization failed: {str(e)}. Using simple tokenization instead.")
        # Simple fallback tokenizer
        import re
        return re.findall(r'\b\w+\b', text.lower())

# Ensure NLTK resources are downloaded first thing
download_nltk_resources()

# Set download directory explicitly (optional, NLTK usually manages this)
# nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
# os.makedirs(nltk_data_dir, exist_ok=True)
# nltk.data.path.append(nltk_data_dir)

# Try to import Groq, fall back to OpenAI if Groq is not available
try:
    from langchain_groq import ChatGroq
    USE_GROQ = True
    # Ensure API key is set if using Groq
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY environment variable not set. Groq functionality might fail.")
except ImportError:
    from langchain_openai import ChatOpenAI
    USE_GROQ = False
    # Ensure API key is set if using OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY environment variable not set. OpenAI functionality might fail.")


warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Irish CHGFs Analysis Dashboard",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for white background, black text, and improved UI
st.markdown("""
<style>
    /* Global reset for consistent appearance */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* White background and black text */
    .main, .block-container, .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Make page stretch across 16:9 screen */
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100% !important;
    }

    /* Make all text black */
    p, li, div, label, span, .stMarkdown, .stText, table, th, td {
        color: #000000 !important;
    }

    /* Header styling */
    h1 {
        color: #103778 !important;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 2px solid #103778;
        margin-bottom: 20px;
    }

    h2 {
        color: #103778 !important;
        font-weight: 600;
        padding-bottom: 5px;
        border-bottom: 1px solid #e6e6e6;
        margin-bottom: 15px;
    }

    h3 {
        color: #103778 !important;
        font-weight: 500;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Improved sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e6e6e6;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] .stTitle {
        color: #103778 !important;
        font-weight: 600;
        padding-left: 1rem;
    }

    /* Button styling */
    .stButton > button {
        background-color: #103778 !important;
        color: white !important;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #1e56a0 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Card styling */
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
        transition: all 0.3s;
        border-left: 4px solid #103778;
        margin-bottom: 1rem;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Improve dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    .stDataFrame [data-testid="stDataFrameResizable"] {
        border: none !important;
    }

    .stDataFrame th {
        background-color: #f1f5f9 !important;
        color: #000000 !important;
        font-weight: 600;
        text-align: left;
        padding: 10px !important;
    }

    .stDataFrame td {
        color: #000000 !important;
        padding: 8px !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f9fa;
        padding: 0px 10px;
        border-radius: 8px 8px 0 0;
        border: 1px solid #e6e6e6;
        border-bottom: none;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        border: none;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #000000 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #103778 !important;
        color: white !important;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: #ffffff;
        border-radius: 0 0 8px 8px;
        border: 1px solid #e6e6e6;
        border-top: none;
        padding: 20px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 4px;
        padding: 10px !important;
        font-weight: 500;
        color: #000000 !important;
        border: 1px solid #e6e6e6;
    }

    .streamlit-expanderContent {
        background-color: #ffffff;
        border-radius: 0 0 4px 4px;
        border: 1px solid #e6e6e6;
        border-top: none;
        padding: 15px;
    }

    /* Chart containers */
    .chart-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e6e6e6;
    }

    /* Chat styling */
    .chat-container {
        border-radius: 10px;
        margin-bottom: 10px;
        padding: 15px;
    }

    .user-message {
        background-color: #e1f5fe;
        border-left: 5px solid #039be5;
    }

    .bot-message {
        background-color: #f0f4c3;
        border-left: 5px solid #afb42b;
    }

    /* Inputs styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox > div > div, .stMultiSelect > div > div {
        border-radius: 4px;
        border: 1px solid #cbd5e0;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #103778;
        box-shadow: 0 0 0 2px rgba(16, 55, 120, 0.2);
    }

    /* Make the chat input area larger */
    .stTextArea textarea {
        min-height: 100px !important;
    }

    /* Improved section headers */
    .section-header {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 5px solid #103778;
    }

    /* Company cards */
    .company-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-top: 4px solid #4e8d7c;
    }

    /* App container */
    .app-container {
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Show NLTK paths to verify
try:
    nltk.data.find('tokenizers/punkt')
    st.success("✅ Punkt tokenizer is available!")
except LookupError as e:
    st.error(f"❌ Punkt tokenizer is NOT available: {str(e)}")
    st.info("Attempting to download NLTK resources...")
    download_nltk_resources() # Attempt download again if missing

try:
    nltk.data.find('corpora/stopwords')
    st.success("✅ Stopwords corpus is available!")
except LookupError as e:
    st.error(f"❌ Stopwords corpus is NOT available: {str(e)}")
    st.info("Attempting to download NLTK resources...")
    download_nltk_resources() # Attempt download again if missing

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None

if 'competitors_found' not in st.session_state:
    st.session_state.competitors_found = []

# Function to load data
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
        else:
            # Attempt to load default file if it exists
            default_file = 'ireland_cleaned_CHGF.xlsx'
            if os.path.exists(default_file):
                df = pd.read_excel(default_file)
                st.info(f"Loaded default file: {default_file}")
            else:
                st.warning("Default file 'ireland_cleaned_CHGF.xlsx' not found. Please upload a file.")
                return pd.DataFrame() # Return empty DataFrame if no file

        # Data cleaning and preparation
        df.columns = df.columns.str.strip() # Remove leading/trailing whitespace from column names
        required_cols = ['Company Name']
        if not all(col in df.columns for col in required_cols):
             st.error(f"Missing required columns. Ensure the file contains at least: {', '.join(required_cols)}")
             return pd.DataFrame()

        # Fill missing Topic values with "Uncategorized"
        if 'Topic' in df.columns:
            df['Topic'] = df['Topic'].fillna('Uncategorized')
        else:
            st.warning("Column 'Topic' not found. Topic-based features will be limited.")
            df['Topic'] = 'Uncategorized' # Create the column if missing

        # Add current year for company age calculation if "Founded Year" exists
        if 'Founded Year' in df.columns:
            # Attempt to convert to numeric, coercing errors to NaN
            df['Founded Year'] = pd.to_numeric(df['Founded Year'], errors='coerce')
            # Drop rows where conversion failed (or handle differently if needed)
            df.dropna(subset=['Founded Year'], inplace=True)
            df['Founded Year'] = df['Founded Year'].astype(int) # Convert valid years to int

            current_year = datetime.datetime.now().year
            df['Company Age'] = current_year - df['Founded Year']
            df['Company Age'] = df['Company Age'].apply(lambda x: max(0, x)) # Ensure age is not negative
        else:
             st.warning("Column 'Founded Year' not found. Age-based features will be unavailable.")

        # Ensure Description column exists and is string
        if 'Description' in df.columns:
             df['Description'] = df['Description'].fillna('').astype(str)
        else:
            st.warning("Column 'Description' not found. Description-based features (Word Clouds, Topic Modeling, Competitor Analysis) will be unavailable.")
            df['Description'] = '' # Create empty Description column

        # Ensure City column exists
        if 'City' not in df.columns:
            st.warning("Column 'City' not found. City-based filters and charts will be unavailable.")
            df['City'] = 'Unknown'

        return df

    except FileNotFoundError:
        st.error(f"Error: Default file 'ireland_cleaned_CHGF.xlsx' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return pd.DataFrame()


# Function to create downloadable link with improved styling
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none;background-color:#103778;color:white;padding:8px 12px;border-radius:4px;font-weight:500;display:inline-block;margin-top:10px;">{text} 📥</a>'
    return href

# Topic modeling with BERTopic
@st.cache_resource
def run_bertopic(texts, n_topics=10):
    try:
        # Initialize sentence transformer model
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize BERTopic
        vectorizer = CountVectorizer(stop_words="english")
        topic_model = BERTopic(
            embedding_model=sentence_model,
            vectorizer_model=vectorizer,
            nr_topics=n_topics,
            calculate_probabilities=True, # Needed for some visualizations if used later
            verbose=True
        )

        # Fit the model
        topics, probs = topic_model.fit_transform(texts)

        return topic_model, topics
    except Exception as e:
        st.error(f"Error during BERTopic modeling: {e}")
        return None, None

# Topic modeling with CorEx
@st.cache_resource
def run_corex(texts, n_topics=10):
    try:
        # Preprocess texts
        vectorizer = CountVectorizer(stop_words='english', max_features=5000, token_pattern=r'\b[a-zA-Z]{3,}\b') # Adjusted token pattern
        X = vectorizer.fit_transform(texts)

        # Check if vocabulary is empty
        if X.shape[1] == 0:
            st.error("No valid words found after preprocessing for CorEx. Check descriptions or adjust preprocessing.")
            return None, None, None

        words = vectorizer.get_feature_names_out()

        # Train CorEx topic model
        topic_model = ct.Corex(n_hidden=n_topics, seed=42) # Added seed for reproducibility
        topic_model.fit(X, words=words)

        # Get topics
        topics = [topic_model.get_topics(topic=i, n_words=10) for i in range(n_topics)]

        return topic_model, topics, words
    except Exception as e:
        st.error(f"Error during CorEx modeling: {e}")
        return None, None, None


# Function to preprocess text
@st.cache_data
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to generate wordcloud
def generate_wordcloud(text, title=None):
    if not text or not text.strip():
         st.info("Cannot generate word cloud from empty text.")
         return None
    try:
        from nltk.corpus import stopwords
        stopwords_set = set(stopwords.words('english'))
    except LookupError:
        st.warning("NLTK stopwords not found. Word cloud might include common words. Attempting download...")
        download_nltk_resources()
        stopwords_set = set(stopwords.words('english')) # Try again


    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        colormap='viridis', # Using a more visually appealing colormap
        stopwords=stopwords_set,
        min_font_size=10,
        max_font_size=150,
        random_state=42,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold', color='#103778')

    plt.tight_layout()
    return fig

# Function to create city distribution pie chart
def create_city_pie_chart(df):
    if 'City' not in df.columns or df['City'].isnull().all():
        st.info("No valid city data available for pie chart.")
        return None

    # Count city occurrences (handle potential non-string types)
    city_counts = df['City'].astype(str).value_counts()

    # For better visualization, limit to top 10 cities and group the rest as "Others"
    if len(city_counts) > 10:
        top_cities = city_counts.head(10)
        others_count = city_counts[10:].sum()
        # Ensure 'Others' is handled correctly even if it exists in top 10
        if 'Others' in top_cities:
             others_count += top_cities['Others']
             top_cities = top_cities.drop('Others')

        # Use pd.concat instead of deprecated append
        city_counts = pd.concat([top_cities, pd.Series([others_count], index=['Others'])])


    # Create pie chart
    fig = px.pie(
        values=city_counts.values,
        names=city_counts.index,
        title='Company Distribution by City',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4
    )

    fig.update_layout(
        title_font=dict(size=20, color='#103778'),
        font=dict(color='#000000', size=14),
        legend_title_font=dict(color='#000000'),
        legend_font=dict(color='#000000'),
        margin=dict(t=50, b=20, l=20, r=20)
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(size=14, color='white'),
        marker=dict(line=dict(color='#ffffff', width=2))
    )

    return fig

# Function to create company age distribution chart
def create_company_age_chart(df):
    if 'Company Age' not in df.columns or df['Company Age'].isnull().all():
        st.info("No valid company age data available for distribution chart.")
        return None

    # Create age groups for better visualization
    try:
        df['Age Group'] = pd.cut(
            df['Company Age'],
            bins=[0, 3, 5, 10, 15, 20, float('inf')], # Use infinity for the last bin edge
            labels=['0-3 years', '3-5 years', '5-10 years', '10-15 years', '15-20 years', '20+ years'],
            right=False # Make bins inclusive of the lower bound: [0, 3), [3, 5), etc.
        )
    except ValueError as e:
        st.error(f"Error creating age groups: {e}. Check 'Company Age' data.")
        return None


    age_distribution = df['Age Group'].value_counts().sort_index()

    # Create bar chart
    fig = px.bar(
        x=age_distribution.index.astype(str), # Ensure index is string for plotting
        y=age_distribution.values,
        color=age_distribution.index.astype(str),
        labels={'x': 'Company Age', 'y': 'Number of Companies'},
        title='Company Age Distribution',
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        title_font=dict(size=20, color='#103778'),
        font=dict(color='#000000', size=14),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        margin=dict(t=50, b=20, l=20, r=20)
    )

    return fig

# Function to create foundation year timeline
def create_foundation_year_timeline(df):
    if 'Founded Year' not in df.columns or df['Founded Year'].isnull().all():
         st.info("No valid 'Founded Year' data available for timeline chart.")
         return None

    # Group by foundation year
    yearly_counts = df['Founded Year'].value_counts().sort_index()

    # Filter out potential invalid years if necessary (e.g., future years or too far past)
    current_year = datetime.datetime.now().year
    yearly_counts = yearly_counts[(yearly_counts.index > 1900) & (yearly_counts.index <= current_year)]

    if yearly_counts.empty:
        st.info("No valid foundation years found within a reasonable range (1901-present).")
        return None

    # Create line chart
    fig = px.line(
        x=yearly_counts.index,
        y=yearly_counts.values,
        markers=True,
        labels={'x': 'Year', 'y': 'Number of Companies Founded'},
        title='Company Foundations Over Time',
    )

    # Add area under the line
    fig.add_traces(
        go.Scatter(
            x=yearly_counts.index,
            y=yearly_counts.values,
            fill='tozeroy',
            fillcolor='rgba(16, 55, 120, 0.2)',
            line=dict(color='rgba(0, 0, 0, 0)'),
            showlegend=False
        )
    )

    fig.update_layout(
        title_font=dict(size=20, color='#103778'),
        font=dict(color='#000000', size=14),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, type='category'), # Treat year as category if large gaps exist
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        margin=dict(t=50, b=20, l=20, r=20)
    )

    fig.update_traces(
        line=dict(color='#103778', width=3),
        marker=dict(size=8, color='#103778')
    )

    return fig

# Function to setup RAG for competitor analysis
def setup_rag_for_competitor_analysis(df):
    if df.empty or 'Description' not in df.columns or df['Description'].isnull().all():
        st.warning("Cannot set up competitor analysis without company descriptions.")
        return None
    try:
        # Create a list of documents from the dataframe
        documents = []

        for idx, row in df.iterrows():
            # Basic content: Name and Description (if available)
            content = f"Company Name: {row.get('Company Name', 'N/A')}\n"
            if pd.notna(row.get('Description')):
                content += f"Description: {row['Description']}\n"

            # Add other relevant columns dynamically
            for col in ['Topic', 'City', 'Founded Year']:
                if col in df.columns and pd.notna(row.get(col)):
                    # Use consistent naming as in find_potential_competitors parsing
                    key_name = 'Industry/Topic' if col == 'Topic' else col
                    content += f"{key_name}: {row[col]}\n"

            # Add any other non-empty columns
            other_cols = [c for c in df.columns if c not in ['Company Name', 'Description', 'Topic', 'City', 'Founded Year', 'Company Age', 'Age Group'] and pd.notna(row.get(c))]
            for col in other_cols:
                 content += f"{col}: {row[col]}\n"


            doc = Document(page_content=content.strip(), metadata={"source": idx, "company": row.get('Company Name', 'N/A')})
            documents.append(doc)

        if not documents:
            st.error("No valid documents could be created from the DataFrame for RAG.")
            return None

        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Initialize the LLM (ChatGroq or ChatOpenAI)
        llm = None
        api_key_found = False
        if USE_GROQ:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if groq_api_key:
                try:
                    llm = ChatGroq(
                        model="mixtral-8x7b-32768", # Correct model name for Groq
                        api_key=groq_api_key,
                        temperature=0.7 # Add temperature for less deterministic answers
                    )
                    api_key_found = True
                    st.sidebar.success("Groq LLM Initialized.")
                except Exception as e:
                    st.sidebar.warning(f"Failed to initialize Groq: {str(e)}. Trying OpenAI.")
            else:
                 st.sidebar.warning("Groq API Key not found. Trying OpenAI.")

        if llm is None: # Fallback to OpenAI if Groq failed or wasn't selected
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                 try:
                    llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        api_key=openai_api_key,
                        temperature=0.7
                    )
                    api_key_found = True
                    st.sidebar.success("OpenAI LLM Initialized.")
                 except Exception as e:
                     st.sidebar.error(f"Failed to initialize OpenAI: {str(e)}")
            else:
                st.sidebar.error("No API Key found for Groq or OpenAI. RAG system cannot be initialized.")
                return None # Cannot proceed without an LLM


        # Create retrieval chain
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 most similar docs
            return_source_documents=True,
            # You might want to customize the prompt for better competitor analysis
            # combine_docs_chain_kwargs={"prompt": custom_prompt}
        )

        return retrieval_chain

    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        return None

# Function to find potential competitors
def find_potential_competitors(company_name, company_description, retrieval_chain):
    if retrieval_chain is None:
        st.error("Competitor analysis system (RAG) is not initialized.")
        return "Competitor analysis system not available.", []
    try:
        # More specific query for competitor analysis
        query = f"""
        Identify and analyze potential competitors for the company "{company_name}".
        Company Details:
        {company_description}

        Based *only* on the provided database of Irish CHGFs, find the top 3-5 companies that are most similar in terms of their business description, industry/sector, and target market.
        For each potential competitor identified from the database, provide:
        1. Company Name
        2. Description
        3. Industry/Topic
        4. A brief explanation of why they are considered a potential competitor based on the similarities.

        Do not include the input company "{company_name}" in the list of competitors.
        Format the output clearly. Start with a summary sentence and then list the competitors.
        """

        # Use invoke for newer Langchain versions if needed, call works too
        # result = retrieval_chain.invoke({"question": query, "chat_history": []})
        result = retrieval_chain({"question": query, "chat_history": []})


        # Extract source documents (competitors) and filter out the input company
        competitors = []
        if "source_documents" in result:
            seen_companies = set([company_name.lower()]) # Keep track to avoid duplicates and self-listing
            for doc in result["source_documents"]:
                company_info = {}
                lines = doc.page_content.strip().split('\n')
                comp_name_found = None
                for line in lines:
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        # Standardize keys parsed from the document content
                        key = key.strip()
                        if key == "Company Name":
                            comp_name_found = value.strip()
                        company_info[key] = value.strip()

                # Add if it's a valid competitor and not the input company itself
                if comp_name_found and comp_name_found.lower() not in seen_companies:
                    if 'Company Name' in company_info: # Ensure name was properly parsed
                        competitors.append(company_info)
                        seen_companies.add(comp_name_found.lower())


        # If the LLM response includes the analysis, use it. Otherwise, generate a summary.
        llm_answer = result.get("answer", "Analysis could not be generated.")

        return llm_answer, competitors

    except Exception as e:
        st.error(f"Error finding competitors: {str(e)}")
        return "An error occurred while analyzing competitors. Please check the RAG setup and API keys.", []

# Main application
def main():
    # App title and header with logo
    col1, col2 = st.columns([1, 6], gap="small")

    with col1:
        # Create a simplified Irish flag icon
        irish_flag = """
        <div style="background-color: white; padding: 5px; border-radius: 5px; text-align: center; height: 60px; display: flex; align-items: center; justify-content: center;">
            <div style="display: flex; height: 40px; width: 60px; border: 1px solid #ccc;">
                <div style="background-color: #169b62; flex: 1;"></div>
                <div style="background-color: white; flex: 1;"></div>
                <div style="background-color: #ff883e; flex: 1;"></div>
            </div>
        </div>
        """
        st.markdown(irish_flag, unsafe_allow_html=True)

    with col2:
        st.title("Irish Consistent High Growth Firms (CHGFs) Analysis") # Simplified title


    # --- File Uploader and Data Loading ---
    st.sidebar.markdown("## 📂 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File (Optional)", type=['xlsx', 'xls'])

    # Load data - either from upload or default
    df = load_data(uploaded_file)

    # --- Main App Logic ---
    if df.empty:
        st.markdown("""
        <div style="background-color: #fffbeb; padding: 20px; border-radius: 8px; border: 1px solid #facc15; margin-top: 20px;">
            <h3 style="color: #b45309; margin-top: 0;">Waiting for Data</h3>
            <p>Please upload an Excel file containing Irish CHGFs data using the sidebar, or place 'ireland_cleaned_CHGF.xlsx' in the same directory as the script.</p>
            <p>Expected columns: 'Company Name'. Optional columns: 'Description', 'Topic', 'City', 'Founded Year'.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop() # Stop execution if no data


    # --- Initialize RAG System ---
    # Attempt initialization only if not already done and data is loaded
    if 'retrieval_chain' not in st.session_state or st.session_state.retrieval_chain is None:
         if 'Description' in df.columns and not df['Description'].isnull().all(): # Check if needed column exists
            with st.spinner("Setting up competitor analysis system..."):
                st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)
         else:
            st.sidebar.warning("Competitor Analysis requires 'Description' column. System not initialized.")
            st.session_state.retrieval_chain = None # Ensure it's None


    # --- Sidebar Filters ---
    st.sidebar.markdown("## 📊 Global Filters")

    df_filtered = df.copy() # Start with a copy of the full dataframe

    # Topic filter
    if 'Topic' in df.columns:
        all_topics = sorted([topic for topic in df['Topic'].unique() if pd.notna(topic)])
        if all_topics:
            selected_topics = st.sidebar.multiselect(
                "Filter by Topic",
                options=all_topics,
                default=[]
            )
            if selected_topics:
                df_filtered = df_filtered[df_filtered['Topic'].isin(selected_topics)]

    # City filter
    if 'City' in df.columns:
         all_cities = sorted([city for city in df['City'].unique() if pd.notna(city)])
         if all_cities:
            selected_cities = st.sidebar.multiselect(
                "Filter by City",
                options=all_cities,
                default=[]
            )
            if selected_cities:
                df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]

    # Company age filter
    if 'Company Age' in df.columns:
         min_age_data = int(df_filtered['Company Age'].min()) if not df_filtered.empty else 0
         max_age_data = int(df_filtered['Company Age'].max()) if not df_filtered.empty else 100
         if max_age_data >= min_age_data: # Ensure valid range
            age_range = st.sidebar.slider(
                "Company Age Range (years)",
                min_value=min_age_data,
                max_value=max_age_data,
                value=(min_age_data, max_age_data)
            )
            df_filtered = df_filtered[
                (df_filtered['Company Age'] >= age_range[0]) &
                (df_filtered['Company Age'] <= age_range[1])
            ]
         else:
             st.sidebar.info("Not enough age data variation for slider.")


    # Company name search
    company_search = st.sidebar.text_input("Search by Company Name")
    if company_search:
        df_filtered = df_filtered[df_filtered['Company Name'].str.contains(company_search, case=False, na=False)]

    # Show filtered data count
    st.sidebar.markdown(f"""
    <div style="background-color: #eef2ff; padding: 10px; border-radius: 5px; margin-top: 15px; text-align: center;">
        <span style="font-weight: 600;">Showing {len(df_filtered)}</span> out of
        <span style="font-weight: 600;">{len(df)}</span> companies
    </div>
    """, unsafe_allow_html=True)

    # Download filtered data
    if not df_filtered.empty:
        st.sidebar.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
        st.sidebar.markdown(
            get_download_link(df_filtered, 'filtered_irish_chgfs.csv', 'Download Filtered Data'),
            unsafe_allow_html=True
        )
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
    else:
        st.sidebar.info("No companies match the current filters.")


    # --- Main Content Tabs ---
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #103778;">
        <p style="margin: 0; font-size: 16px;">
        Interactive analysis of Irish companies identified as
        <strong>Consistent High Growth Firms (CHGFs)</strong>. Explore data, analyze topics, and find potential competitors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_titles = [
        "📊 Dashboard",
        "🔍 Company Explorer",
        "🏷️ Topic Analysis",
        "🧠 Advanced Topic Modeling",
        "🥇 Competitor Analysis"
    ]
    tabs = st.tabs(tab_titles)


    # == Dashboard Overview Tab ==
    with tabs[0]:
        st.header("Dashboard Overview")

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Companies", f"{len(df)}")
            st.markdown("<p style='color: #666; font-size: 14px;'>Total companies in dataset</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'Topic' in df.columns:
                st.metric("Number of Topics", f"{df['Topic'].nunique()}") # More robust count
                st.markdown("<p style='color: #666; font-size: 14px;'>Distinct business categories</p>", unsafe_allow_html=True)
            else:
                st.metric("Number of Topics", "N/A")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'City' in df.columns:
                st.metric("Cities Represented", f"{df['City'].nunique()}")
                st.markdown("<p style='color: #666; font-size: 14px;'>Unique locations</p>", unsafe_allow_html=True)
            else:
                st.metric("Cities Represented", "N/A")
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'Topic' in df.columns and len(df) > 0:
                companies_with_topic = len(df[df['Topic'] != 'Uncategorized'])
                topic_coverage = round((companies_with_topic / len(df)) * 100, 1) if len(df) > 0 else 0
                st.metric("Topic Coverage", f"{topic_coverage}%")
                st.markdown("<p style='color: #666; font-size: 14px;'>Companies with assigned topics</p>", unsafe_allow_html=True)
            else:
                st.metric("Topic Coverage", "N/A")
            st.markdown("</div>", unsafe_allow_html=True)

        # Charts
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Topic Distribution Chart
            if 'Topic' in df.columns and df['Topic'].nunique() > 1: # Check if topic column exists and has variety
                st.subheader("Topic Distribution")
                topic_counts = df['Topic'].value_counts().reset_index()
                topic_counts.columns = ['Topic', 'Count']

                 # Limit number of topics shown for clarity
                max_topics_chart = 15
                if len(topic_counts) > max_topics_chart:
                     top_topics = topic_counts.head(max_topics_chart)
                     other_count = topic_counts[max_topics_chart:]['Count'].sum()
                     other_row = pd.DataFrame([{'Topic': 'Others', 'Count': other_count}])
                     topic_counts = pd.concat([top_topics, other_row], ignore_index=True)


                fig = px.bar(
                    topic_counts,
                    x='Topic',
                    y='Count',
                    color='Topic',
                    title='Distribution of Companies by Topic',
                    height=500,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='white',
                    title_font=dict(size=20, color='#103778'),
                    font=dict(color='#000000'),
                    legend_title_font=dict(color='#000000'),
                    xaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                    yaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000'))
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Topic distribution chart requires 'Topic' column with multiple categories.")

            # Company Age Chart (if available)
            if 'Company Age' in df.columns:
                st.subheader("Company Age Distribution")
                age_chart = create_company_age_chart(df.copy()) # Pass copy to avoid modifying original df
                if age_chart:
                    st.plotly_chart(age_chart, use_container_width=True)


        with chart_col2:
            # City Distribution Chart (if available)
            if 'City' in df.columns:
                st.subheader("City Distribution")
                city_chart = create_city_pie_chart(df)
                if city_chart:
                    st.plotly_chart(city_chart, use_container_width=True)

             # Foundation Year Timeline
            if 'Founded Year' in df.columns:
                st.subheader("Company Foundations Over Time")
                timeline_chart = create_foundation_year_timeline(df)
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)


        # Display sample data
        st.markdown("---")
        st.subheader("Sample Data Preview")
        if len(df) > 0:
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("The dataset is empty.")


    # == Company Explorer Tab ==
    with tabs[1]:
        st.header("Company Explorer")

        # Use the globally filtered dataframe 'df_filtered' here
        if df_filtered.empty:
             st.warning("No companies match the current filter criteria.")
        else:
            # Advanced search and filters specific to this tab (optional)
            # For now, we use the sidebar filters applied to df_filtered
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("""
                <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h3 style="color: #103778; margin-top: 0;">Display Options</h3>
                </div>
                """, unsafe_allow_html=True)

                # Sort options based on available columns in the filtered data
                sort_options = [col for col in ["Company Name", "Topic", "Founded Year", "City", "Company Age"] if col in df_filtered.columns]
                sort_by = st.selectbox("Sort by", sort_options)

                st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
                sort_order_asc = st.radio("Sort order", ["Ascending", "Descending"]) == "Ascending"
                st.markdown("</div>", unsafe_allow_html=True)

                # Apply sorting
                if sort_by in df_filtered.columns:
                     df_display = df_filtered.sort_values(by=sort_by, ascending=sort_order_asc, na_position='last')
                else:
                     df_display = df_filtered # Default sort if column somehow missing

            with col2:
                st.markdown(f"""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #103778;">
                    <h3 style="color: #103778; margin-top: 0;">Displaying {len(df_display)} Companies</h3>
                </div>
                """, unsafe_allow_html=True)

                # Pagination
                st.markdown("<div style='background-color: #f8fafc; padding: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
                items_per_page = st.slider("Companies per page", 5, 50, 10)
                total_pages = max(1, (len(df_display) + items_per_page - 1) // items_per_page)
                current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

                start_idx = (current_page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(df_display))

                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <p style="font-weight: 500;">Showing {start_idx + 1} to {end_idx} of {len(df_display)} entries</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Display company cards
            st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
            for idx, row in df_display.iloc[start_idx:end_idx].iterrows():
                 with st.expander(f"{row.get('Company Name', 'N/A')}", expanded=False):
                    exp_col1, exp_col2 = st.columns([1, 3])

                    with exp_col1:
                        # Display Topic
                        st.markdown("<p style='font-weight: 600;'>Topic:</p>", unsafe_allow_html=True)
                        topic = row.get('Topic', "N/A")
                        st.markdown(f"""
                        <div style="background-color: #e0f2fe; padding: 8px; border-radius: 4px; text-align: center; margin-bottom: 10px;">
                            <p style="margin: 0; color: #0c4a6e; font-weight: 500;">{topic}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display City
                        if 'City' in row and pd.notna(row['City']):
                            st.markdown("<p style='font-weight: 600;'>City:</p>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background-color: #e0f7fa; padding: 8px; border-radius: 4px; text-align: center; margin-bottom: 10px;">
                                <p style="margin: 0; color: #006064; font-weight: 500;">{row['City']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Display Founded Year & Age
                        if 'Founded Year' in row and pd.notna(row['Founded Year']):
                            st.markdown("<p style='font-weight: 600;'>Founded / Age:</p>", unsafe_allow_html=True)
                            age = row.get('Company Age', 'N/A')
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; text-align: center;">
                                <p style="margin: 0; color: #1b5e20; font-weight: 500;">{int(row['Founded Year'])} ({age} yrs)</p>
                            </div>
                            """, unsafe_allow_html=True)


                    with exp_col2:
                        # Display Description
                        st.markdown("<p style='font-weight: 600;'>Description:</p>", unsafe_allow_html=True)
                        description = row.get('Description', "No description available.")
                        if not description: description = "No description available." # Handle empty string case
                        st.markdown(f"""
                        <div style="background-color: #f8fafc; padding: 10px; border-radius: 4px; border-left: 3px solid #103778; margin-bottom: 15px;">
                            <p style="margin: 0;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display other columns dynamically
                        other_cols_to_show = [c for c in df_display.columns if c not in ['Company Name', 'Description', 'Topic', 'City', 'Founded Year', 'Company Age', 'Age Group'] and pd.notna(row.get(c))]
                        if other_cols_to_show:
                            st.markdown("<p style='font-weight: 600;'>Additional Information:</p>", unsafe_allow_html=True)
                            details = {col: row[col] for col in other_cols_to_show}
                            st.json(details, expanded=False) # Display as JSON in collapsed section


            st.markdown("</div>", unsafe_allow_html=True)


    # == Topic Analysis Tab ==
    with tabs[2]:
        st.header("Topic Analysis")

        if 'Topic' in df.columns and 'Description' in df.columns:
            all_topics = sorted([topic for topic in df['Topic'].unique() if pd.notna(topic)])
            if not all_topics:
                 st.warning("No valid topics found in the 'Topic' column.")
            else:
                st.markdown("""
                <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: #103778; margin-top: 0; margin-bottom: 10px;">Select Topic to Analyze</h3>
                </div>
                """, unsafe_allow_html=True)

                topic_for_analysis = st.selectbox(
                    "", # No label needed due to header above
                    options=all_topics,
                    key="topic_analysis_select" # Add unique key
                )

                if topic_for_analysis:
                    topic_companies = df[df['Topic'] == topic_for_analysis]

                    st.markdown(f"""
                    <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #103778;">
                        <h2 style="color: #103778; margin-top: 0;">Analysis of Topic: {topic_for_analysis}</h2>
                        <p style="font-size: 16px; margin-bottom: 0;">
                            Number of companies in this topic: <strong>{len(topic_companies)}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Companies in this Topic", expanded=False):
                         st.dataframe(topic_companies[['Company Name', 'Description']], use_container_width=True)
                         st.markdown(
                            get_download_link(topic_companies, f'irish_chgfs_{topic_for_analysis}.csv', f'Download {topic_for_analysis} Companies'),
                            unsafe_allow_html=True
                         )

                    # Define combined_description for the selected topic
                    combined_description = ' '.join(topic_companies['Description'].fillna('').astype(str))

                    # Create word cloud for this topic
                    st.subheader("Topic Word Cloud")
                    if combined_description.strip():
                        try:
                            wordcloud_fig = generate_wordcloud(combined_description, f"Word Cloud for {topic_for_analysis}")
                            if wordcloud_fig:
                                st.pyplot(wordcloud_fig)
                        except Exception as e:
                            st.error(f"Error generating word cloud: {str(e)}")
                    else:
                        st.info("No description data available for word cloud generation for this topic.")

                    # Common words analysis
                    st.subheader("Most Common Terms")
                    if combined_description.strip():
                        try:
                            from nltk.corpus import stopwords
                            stop_words = set(stopwords.words('english'))

                            # Tokenize and preprocess
                            tokens = safe_tokenize(preprocess_text(combined_description))

                            # Filter words
                            words = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]

                            if words:
                                # Calculate word frequencies
                                word_freq = nltk.FreqDist(words)
                                top_words = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Frequency'])

                                # Display as bar chart
                                fig = px.bar(
                                    top_words,
                                    x='Word',
                                    y='Frequency',
                                    color='Word',
                                    title=f'Most Common Terms in Topic: {topic_for_analysis}',
                                    height=500,
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                fig.update_layout(
                                    plot_bgcolor='white',
                                    title_font=dict(size=18, color='#103778'),
                                    font=dict(color='#000000'),
                                    xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
                                    yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12))
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                 st.info("No common terms found for this topic after filtering.")

                        except Exception as e:
                            st.error(f"Error analyzing common terms: {str(e)}")
                    else:
                        st.info("No description data available for term analysis for this topic.")


                    # Topic similarity analysis (optional - can be computationally intensive)
                    # Consider adding a button or checkbox to trigger this if needed
                    # ... (code for similarity matrix) ...

        else:
            st.warning("Topic analysis requires both 'Topic' and 'Description' columns in the dataset.")


    # == Advanced Topic Modeling Tab ==
    with tabs[3]:
        st.header("Advanced Topic Modeling")

        if 'Description' in df.columns:
            descriptions = df['Description'].dropna().astype(str).tolist()
            descriptions = [desc for desc in descriptions if desc.strip() and len(desc.split()) > 5] # Filter short/empty descriptions

            if descriptions:
                st.markdown("""
                <p>Run advanced topic modeling algorithms (BERTopic or CorEx) on the company descriptions to discover underlying themes.</p>
                """, unsafe_allow_html=True)

                modeling_method = st.radio(
                    "Select Topic Modeling Method",
                    ["BERTopic", "CorEx"],
                    key="topic_model_method"
                )

                num_topics = st.slider(
                    "Number of Topics to Generate",
                    min_value=5, max_value=30, value=10, step=1,
                    key="num_topics_slider"
                )

                run_modeling = st.button(f"Run {modeling_method} Topic Modeling", key=f"run_{modeling_method}")

                if run_modeling:
                    with st.spinner(f"Running {modeling_method} topic modeling. This may take a few minutes..."):
                        if modeling_method == "BERTopic":
                            topic_model, topics = run_bertopic(descriptions, num_topics)
                            if topic_model and topics is not None:
                                st.success("BERTopic modeling complete!")
                                st.subheader("BERTopic Results")
                                topic_info = topic_model.get_topic_info()
                                st.write("Topic Overview:")
                                st.dataframe(topic_info, use_container_width=True)

                                st.subheader("Top Terms per Topic")
                                for topic_id in topic_info['Topic']:
                                    if topic_id != -1: # Skip outlier topic
                                        topic_terms = topic_model.get_topic(topic_id)
                                        topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].iloc[0]
                                        st.write(f"**{topic_name} (Topic {topic_id})**:")
                                        terms_df = pd.DataFrame(topic_terms, columns=["Term", "Score"])
                                        st.dataframe(terms_df.head(10), use_container_width=True) # Show top 10 terms

                                # Visualization (optional - add if needed)
                                # fig_topics = topic_model.visualize_topics()
                                # st.plotly_chart(fig_topics)

                        elif modeling_method == "CorEx":
                            topic_model, topics, words = run_corex(descriptions, num_topics)
                            if topic_model and topics:
                                st.success("CorEx modeling complete!")
                                st.subheader("CorEx Results")
                                st.write(f"Total Correlation (TC): {topic_model.tc:.2f}")

                                topic_words_list = []
                                for i, topic_w in enumerate(topics):
                                    st.write(f"**Topic {i}**:")
                                    st.write(", ".join([w for w, _ in topic_w]))
                                    topic_words_list.extend([{"Topic": i, "Word": w, "Weight": s} for w, s in topic_w])

                                # Visualization (optional)
                                # You might want to plot TC or topic correlations if needed
                                # fig_corr = px.imshow(topic_model.tcs, title="Topic Correlation (TC)")
                                # st.plotly_chart(fig_corr)
                            elif topic_model is None:
                                # Error handled within run_corex
                                pass


            else:
                st.warning("No valid company descriptions found for topic modeling after filtering.")
        else:
            st.warning("The dataset does not contain a 'Description' column required for advanced topic modeling.")


    # == Competitor Analysis Tab ==
    with tabs[4]:
        st.header("Competitor Analysis Chatbot")

        # Introduction
        st.markdown("""
        <div style="background-color: #f0f9ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0369a1;">
            <h3 style="color: #0369a1; margin-top: 0;">Find Potential Competitors</h3>
            <p>Use AI to identify potential competitors from the loaded CHGFs database based on your company's details.</p>
        </div>
        """, unsafe_allow_html=True)

        # Check if retrieval chain is available
        if st.session_state.retrieval_chain is None:
            st.error("Competitor analysis system is not available. This might be due to missing API keys or lack of 'Description' data in the input file.")
            # Option to retry setup if potentially fixable (e.g., API key entered later)
            # if st.button("Retry RAG Setup"):
            #     with st.spinner("Attempting to set up competitor analysis system..."):
            #         st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)
            #         if st.session_state.retrieval_chain is not None:
            #             st.success("Setup successful!")
            #             st.experimental_rerun()
            #         else:
            #             st.error("Setup failed. Please check data and API keys.")
        else:
            # Company input form
            with st.form("competitor_form"):
                col1_form, col2_form = st.columns([1, 1])

                with col1_form:
                    company_name_input = st.text_input("Your Company Name*", key="comp_ana_name")

                with col2_form:
                    # Use topics from the data if available, otherwise provide generic list
                    industry_options = sorted([t for t in df['Topic'].unique() if t != 'Uncategorized']) if 'Topic' in df.columns else ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail", "Services", "Other"]
                    industry_type_input = st.selectbox(
                        "Your Industry/Sector*",
                        options=[""] + industry_options, # Add blank option
                        key="comp_ana_industry"
                    )

                company_description_input = st.text_area(
                    "Describe your company (business model, products/services, target market)*",
                    height=150,
                    placeholder="E.g., We develop AI-powered financial analytics software for SMEs...",
                    key="comp_ana_desc"
                )

                # --- Advanced Options Removed from form for simplicity, handled in query ---
                # num_results_input = 5 # Default

                submitted = st.form_submit_button("Find Potential Competitors")

            # Process form submission
            if submitted:
                if not company_name_input or not company_description_input or not industry_type_input:
                    st.error("Please fill in all required fields (*) to find competitors.")
                else:
                    with st.spinner("Analyzing potential competitors..."):
                        # Combine inputs for the analysis query
                        full_desc_for_query = (
                            f"Company Name: {company_name_input}\n"
                            f"Industry/Sector: {industry_type_input}\n"
                            f"Description: {company_description_input}"
                        )
                        analysis_text, competitors_found = find_potential_competitors(
                            company_name_input,
                            full_desc_for_query,
                            st.session_state.retrieval_chain
                        )

                        st.session_state.competitors_found = competitors_found # Store found competitors

                        # Display Analysis Results
                        st.markdown("## Competitor Analysis Results")
                        st.markdown(f"<div class='chat-container bot-message'>{analysis_text}</div>", unsafe_allow_html=True)

                        # Display Competitor Cards
                        if competitors_found:
                            st.markdown("### Top Matching Companies from Database")
                            num_to_display = 5 # Let's fix this to 5 for now
                            cols = st.columns(min(3, len(competitors_found[:num_to_display]))) # Max 3 columns

                            for i, competitor in enumerate(competitors_found[:num_to_display]):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    comp_name = competitor.get('Company Name', 'Unknown')
                                    description = competitor.get('Description', 'No description.')
                                    topic = competitor.get('Industry/Topic', 'N/A')

                                    st.markdown(f"""
                                    <div class="company-card" style="border-top-color: #103778;">
                                        <h4 style="color: #103778; margin-top: 0; margin-bottom: 10px;">{comp_name}</h4>
                                        <p style="font-size: 14px; margin-bottom: 5px;"><strong>Industry:</strong> {topic}</p>
                                        <p style="font-size: 14px; max-height: 100px; overflow-y: auto;">{description}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                             # Check if the analysis text mentions inability to find competitors
                            if "no specific competitors" in analysis_text.lower() or "could not identify" in analysis_text.lower():
                                st.info("The analysis indicates no direct competitors were found in the database based on your description.")
                            else:
                                st.info("No competitor details were extracted, but see the analysis above.")


                        # Download report option
                        if competitors_found:
                            competitors_df = pd.DataFrame(competitors_found[:num_to_display]) # Use the displayed number
                            st.markdown("<div style='text-align: left; margin-top: 15px;'>", unsafe_allow_html=True)
                            st.markdown(
                                get_download_link(competitors_df, 'competitor_analysis.csv', 'Download Competitor Details'),
                                unsafe_allow_html=True
                            )
                            st.markdown("</div>", unsafe_allow_html=True)

            # --- Interactive Chat Section ---
            st.markdown("---")
            st.subheader("Interactive Competitor Chat")
            st.markdown("""
            <div style="background-color: #f0f4c3; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #afb42b;">
                <p style="margin: 0;">Ask follow-up questions about the competitors found or the market landscape based on the CHGF data.</p>
            </div>
            """, unsafe_allow_html=True)

            # Display chat history
            chat_display_area = st.container() # Use container for better control if needed
            with chat_display_area:
                for i, message in enumerate(st.session_state.chat_history):
                    role_class = "user-message" if message["role"] == "user" else "bot-message"
                    role_label = "You" if message["role"] == "user" else "AI"
                    st.markdown(f"""<div class="chat-container {role_class}">
                        <p><strong>{role_label}:</strong> {message["content"]}</p>
                    </div>""", unsafe_allow_html=True)


            # Chat input form using st.form
            with st.form(key='chat_form', clear_on_submit=True):
                 chat_input = st.text_area("Ask a question:", key="chat_input_area", height=100)
                 send_button = st.form_submit_button("Send")

            # Handle chat submission outside the form
            if send_button and chat_input:
                st.session_state.chat_history.append({"role": "user", "content": chat_input})

                # Prepare context (e.g., previously found competitors) for the LLM
                context_for_llm = "Recent Competitor Analysis Summary:\n"
                if st.session_state.competitors_found:
                     for comp in st.session_state.competitors_found[:3]: # Limit context length
                         context_for_llm += f"- {comp.get('Company Name', '?')} ({comp.get('Industry/Topic', '?')}): {comp.get('Description', '?')[:100]}...\n"
                else:
                    context_for_llm += "No specific competitors identified in the last analysis.\n"

                # Prepare chat history for Langchain format (list of tuples)
                formatted_chat_history = []
                for msg in st.session_state.chat_history[:-1]: # Exclude the latest user message
                     if msg["role"] == "user":
                         formatted_chat_history.append((msg["content"], "")) # User message
                     elif msg["role"] == "assistant":
                         if formatted_chat_history:
                             # Add assistant response to the last user message tuple
                             last_user_msg, _ = formatted_chat_history[-1]
                             formatted_chat_history[-1] = (last_user_msg, msg["content"])
                         # else: Handle case where bot message is first (shouldn't happen in normal flow)


                # Construct the query for the LLM
                llm_query = f"""
                {context_for_llm}
                Based on the Irish CHGF dataset and the analysis context above, answer the following user question concisely:
                User Question: {chat_input}
                """

                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.retrieval_chain({"question": llm_query, "chat_history": formatted_chat_history})
                        response = result.get("answer", "Sorry, I couldn't process that question.")
                    except Exception as e:
                        response = f"An error occurred: {str(e)}. Please try again."
                        st.error(response)

                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.experimental_rerun() # Rerun to display the new message


            # Clear chat button
            if st.button("Clear Chat History", key="clear_chat"):
                st.session_state.chat_history = []
                st.session_state.competitors_found = [] # Also clear found competitors context
                st.experimental_rerun()


if __name__ == "__main__":
    # Basic check for event loop, common issue with Streamlit and async libraries
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    main()