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
import asyncio # Added for potential event loop issues

# --- NLTK Setup ---
@st.cache_resource
def download_nltk_resources():
    """Downloads necessary NLTK data (punkt, stopwords) if not found."""
    import os
    import nltk
    needed = {'tokenizers/punkt': 'punkt', 'corpora/stopwords': 'stopwords'}
    downloaded_any = False
    for resource_path, download_name in needed.items():
        try:
            nltk.data.find(resource_path)
            print(f"NLTK resource '{download_name}' already downloaded.")
        except LookupError:
            print(f"Downloading NLTK resource '{download_name}'...")
            try:
                nltk.download(download_name, quiet=False)
                print(f"Successfully downloaded '{download_name}'.")
                downloaded_any = True
            except Exception as e:
                st.error(f"Failed to download NLTK resource '{download_name}': {str(e)}")
                st.error("Word tokenization and stopword removal might fail.")
    # Add a success message if downloads happened
    # if downloaded_any:
    #     st.sidebar.success("NLTK resources downloaded.")


# Initial download attempt
download_nltk_resources()

def safe_tokenize(text):
    """Safely tokenizes text, handling potential NLTK errors."""
    if text is None or not isinstance(text, str):
        return []
    try:
        # Ensure resources are available before tokenizing
        nltk.data.find('tokenizers/punkt')
        return nltk.word_tokenize(text)
    except LookupError:
        st.warning("NLTK 'punkt' tokenizer not found during tokenization. Attempting download again...")
        download_nltk_resources()
        try:
            return nltk.word_tokenize(text) # Try again after download
        except Exception as e:
             print(f"NLTK tokenization failed even after download attempt: {str(e)}. Using simple fallback.")
             return re.findall(r'\b\w+\b', text.lower()) # Fallback
    except Exception as e:
        print(f"NLTK tokenization failed: {str(e)}. Using simple fallback.")
        return re.findall(r'\b\w+\b', text.lower()) # Fallback


def safe_get_stopwords():
    """Safely gets NLTK stopwords, handling potential errors."""
    try:
        nltk.data.find('corpora/stopwords')
        return set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        st.warning("NLTK 'stopwords' not found. Attempting download...")
        download_nltk_resources()
        try:
             return set(nltk.corpus.stopwords.words('english')) # Try again
        except Exception as e:
            st.error(f"Could not load stopwords even after download attempt: {e}")
            return set() # Return empty set if fails
    except Exception as e:
        st.error(f"Error loading stopwords: {e}")
        return set() # Return empty set

# --- End NLTK Setup ---


# --- LLM and Environment Variable Setup ---
LLM_INITIALIZED = False
RAG_ENABLED = False
try:
    from langchain_groq import ChatGroq
    USE_GROQ = True
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
         st.sidebar.warning("GROQ_API_KEY environment variable not set.")
except ImportError:
    USE_GROQ = False
    GROQ_API_KEY = None

try:
    from langchain_openai import ChatOpenAI
    USE_OPENAI = True
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY and not (USE_GROQ and GROQ_API_KEY) : # Only warn if Groq isn't usable either
        st.sidebar.warning("OPENAI_API_KEY environment variable not set.")
except ImportError:
    USE_OPENAI = False
    OPENAI_API_KEY = None

# Determine if any LLM can be used for RAG
if (USE_GROQ and GROQ_API_KEY) or (USE_OPENAI and OPENAI_API_KEY):
    RAG_ENABLED = True
else:
    st.sidebar.error("No valid API key found for Groq or OpenAI. Competitor Analysis disabled.")

warnings.filterwarnings('ignore')
# --- End LLM Setup ---


# --- Streamlit Page Config and CSS ---
st.set_page_config(
    page_title="Irish CHGFs Analysis Dashboard",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global reset for consistent appearance */
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    /* White background and black text */
    .main, .block-container, .stApp { background-color: #ffffff !important; color: #000000 !important; }
    /* Page width */
    .block-container { padding-left: 1rem; padding-right: 1rem; max-width: 100% !important; }
    /* Default text color */
    p, li, div, label, span, .stMarkdown, .stText, table, th, td { color: #000000 !important; }
    /* Headers */
    h1 { color: #103778 !important; font-weight: 700; padding-bottom: 10px; border-bottom: 2px solid #103778; margin-bottom: 20px; }
    h2 { color: #103778 !important; font-weight: 600; padding-bottom: 5px; border-bottom: 1px solid #e6e6e6; margin-bottom: 15px; margin-top: 30px; }
    h3 { color: #103778 !important; font-weight: 500; margin-top: 20px; margin-bottom: 10px; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #e6e6e6; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] .stTitle { color: #103778 !important; font-weight: 600; padding-left: 1rem; }
    /* Buttons */
    .stButton > button { background-color: #103778 !important; color: white !important; border-radius: 4px; border: none; padding: 0.5rem 1rem; font-weight: 500; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #1e56a0 !important; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    /* Metric Cards */
    .metric-card { background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); padding: 20px; text-align: center; transition: all 0.3s; border-left: 4px solid #103778; margin-bottom: 1rem; height: 130px; display: flex; flex-direction: column; justify-content: center; }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); }
    .metric-card p { font-size: 0.85rem; color: #666 !important; margin-top: 5px;}
    /* DataFrames */
    .stDataFrame { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stDataFrame [data-testid="stDataFrameResizable"] { border: none !important; }
    .stDataFrame th { background-color: #f1f5f9 !important; color: #000000 !important; font-weight: 600; text-align: left; padding: 10px !important; }
    .stDataFrame td { color: #000000 !important; padding: 8px !important; vertical-align: top;}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #f8f9fa; padding: 0px 10px; border-radius: 8px 8px 0 0; border: 1px solid #e6e6e6; border-bottom: none; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f8f9fa; border-radius: 8px 8px 0 0; border: none; padding: 10px 15px; color: #000000 !important; }
    .stTabs [aria-selected="true"] { background-color: #103778 !important; color: white !important; font-weight: 500; }
    .stTabs [data-baseweb="tab-panel"] { background-color: #ffffff; border-radius: 0 0 8px 8px; border: 1px solid #e6e6e6; border-top: none; padding: 20px; }
    /* Expanders */
    .streamlit-expanderHeader { background-color: #f8fafc; border-radius: 4px; padding: 10px !important; font-weight: 500; color: #000000 !important; border: 1px solid #e6e6e6; }
    .streamlit-expanderContent { background-color: #ffffff; border-radius: 0 0 4px 4px; border: 1px solid #e6e6e6; border-top: none; padding: 15px; }
    /* Chat */
    .chat-container { border-radius: 10px; margin-bottom: 10px; padding: 15px; }
    .user-message { background-color: #e1f5fe; border-left: 5px solid #039be5; }
    .bot-message { background-color: #f0f4c3; border-left: 5px solid #afb42b; }
    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox > div > div, .stMultiSelect > div > div { border-radius: 4px; border: 1px solid #cbd5e0; }
    .stTextInput input:focus, .stTextArea textarea:focus { border-color: #103778; box-shadow: 0 0 0 2px rgba(16, 55, 120, 0.2); }
    .stTextArea textarea { min-height: 100px !important; }
    /* Company Cards (Explorer & Competitor) */
    .company-card { background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-top: 4px solid #4e8d7c; min-height: 150px; /* Ensure cards have some min height */ }
    .company-card h4 { color: #103778; margin-top: 0; margin-bottom: 10px; font-size: 1.1rem;}
    .company-card p { font-size: 0.9rem; margin-bottom: 5px; line-height: 1.4;}
    .company-card strong { font-weight: 600;}
    .company-card .description { max-height: 80px; overflow-y: auto; /* Limit description height */ }
</style>
""", unsafe_allow_html=True)
# --- End CSS ---


# --- Session State Initialization ---
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'competitors_found' not in st.session_state:
    st.session_state.competitors_found = []
# --- End Session State ---


# --- Data Loading Function ---
@st.cache_data # Cache the loaded and processed data
def load_data(uploaded_file=None):
    """Loads data from uploaded file or default, performs cleaning."""
    df = pd.DataFrame() # Initialize empty DataFrame
    try:
        file_source = None
        if uploaded_file:
            file_source = uploaded_file
            st.sidebar.success("Using uploaded file.")
        else:
            default_file = 'ireland_cleaned_CHGF.xlsx'
            if os.path.exists(default_file):
                file_source = default_file
                st.sidebar.info(f"Using default file: {default_file}")
            else:
                # This state is handled before calling load_data in main()
                return pd.DataFrame()

        df = pd.read_excel(file_source)

        # --- Data Cleaning & Preparation ---
        df.columns = df.columns.str.strip() # Clean column names

        # Check essential columns
        if 'Company Name' not in df.columns:
             st.error("Missing required column: 'Company Name'. Cannot proceed.")
             return pd.DataFrame()

        # Handle optional columns gracefully
        if 'Topic' not in df.columns: st.warning("Column 'Topic' not found. Creating default."); df['Topic'] = 'Uncategorized'
        if 'Description' not in df.columns: st.warning("Column 'Description' not found. Creating default."); df['Description'] = ''
        if 'City' not in df.columns: st.warning("Column 'City' not found. Creating default."); df['City'] = 'Unknown'

        # Clean specific columns
        df['Topic'] = df['Topic'].fillna('Uncategorized').astype(str)
        df['Description'] = df['Description'].fillna('').astype(str)
        df['City'] = df['City'].fillna('Unknown').astype(str)
        df['Company Name'] = df['Company Name'].fillna('Unknown').astype(str)

        # Handle 'Founded Year' and calculate 'Company Age'
        if 'Founded Year' in df.columns:
            df['Founded Year'] = pd.to_numeric(df['Founded Year'], errors='coerce')
            df.dropna(subset=['Founded Year'], inplace=True) # Drop rows where year is invalid
            if not df.empty:
                 df['Founded Year'] = df['Founded Year'].astype(int)
                 current_year = datetime.datetime.now().year
                 df['Company Age'] = current_year - df['Founded Year']
                 df['Company Age'] = df['Company Age'].apply(lambda x: max(0, x)) # Ensure non-negative
            else:
                 st.warning("No valid 'Founded Year' entries found after cleaning.")
                 df['Company Age'] = pd.NA # Assign NA if no valid years
        else:
             st.warning("Column 'Founded Year' not found. Age features unavailable.")
             df['Company Age'] = pd.NA # Assign NA if column missing


        return df

    except FileNotFoundError:
        st.error(f"Error: Default file '{default_file}' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        # Consider logging the full error details here for debugging
        # logger.exception("Data loading failed")
        return pd.DataFrame()
# --- End Data Loading ---


# --- Utility Functions ---
def get_download_link(df, filename, text):
    """Generates an HTML download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none;background-color:#103778;color:white;padding:8px 12px;border-radius:4px;font-weight:500;display:inline-block;margin-top:10px;">{text} 📥</a>'

@st.cache_data # Cache preprocessed text
def preprocess_text(text):
    """Basic text preprocessing: lowercase, remove non-alpha, strip whitespace."""
    if pd.isna(text) or not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep only letters and space
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def generate_wordcloud(text, title=None):
    """Generates a WordCloud figure from text."""
    if not text or not text.strip():
         st.info("Cannot generate word cloud from empty text.")
         return None

    stopwords_set = safe_get_stopwords() # Get stopwords safely

    try:
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            colormap='viridis', stopwords=stopwords_set,
            min_font_size=10, max_font_size=120, # Adjusted sizes
            random_state=42, contour_width=1, contour_color='steelblue'
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5)) # Adjusted size
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        if title: ax.set_title(title, fontsize=16, fontweight='bold', color='#103778')
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None
# --- End Utility Functions ---


# --- Plotting Functions ---
@st.cache_data # Cache plot figures based on DataFrame hash
def create_city_pie_chart(df_plot):
    """Creates a Plotly Pie chart for city distribution."""
    if 'City' not in df_plot.columns or df_plot['City'].isnull().all():
        return None
    city_counts = df_plot['City'].astype(str).value_counts()
    if len(city_counts) > 10: # Group small slices
        top_cities = city_counts.head(10)
        others_count = city_counts[10:].sum()
        if 'Others' in top_cities: others_count += top_cities.pop('Others')
        city_counts_display = pd.concat([top_cities, pd.Series([others_count], index=['Others'])])
    else:
        city_counts_display = city_counts

    if city_counts_display.empty: return None

    fig = px.pie(values=city_counts_display.values, names=city_counts_display.index,
                 title='Company Distribution by City', color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
    fig.update_layout(title_font_size=20, font_size=14, margin=dict(t=50, b=20, l=20, r=20), legend_title_text='Cities')
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=12, color='black'), # Black text for Pastel
                      marker=dict(line=dict(color='#ffffff', width=1)))
    return fig

@st.cache_data
def create_company_age_chart(df_plot):
    """Creates a Plotly Bar chart for company age distribution."""
    if 'Company Age' not in df_plot.columns or df_plot['Company Age'].isnull().all():
        return None
    try:
        # Use a copy to avoid SettingWithCopyWarning if df_plot is a slice
        df_temp = df_plot.copy()
        df_temp['Age Group'] = pd.cut(
            df_temp['Company Age'], bins=[0, 3, 5, 10, 15, 20, float('inf')],
            labels=['0-3', '4-5', '6-10', '11-15', '16-20', '21+'], right=False
        )
        age_distribution = df_temp['Age Group'].value_counts().sort_index()
    except Exception as e: # Catch potential errors in cutting
        st.error(f"Could not create age groups: {e}")
        return None

    if age_distribution.empty: return None

    fig = px.bar(x=age_distribution.index.astype(str), y=age_distribution.values,
                 color=age_distribution.index.astype(str), labels={'x': 'Company Age (Years)', 'y': 'Number of Companies'},
                 title='Company Age Distribution', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(title_font_size=20, font_size=14, plot_bgcolor='white', showlegend=False,
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                      margin=dict(t=50, b=20, l=20, r=20))
    return fig

@st.cache_data
def create_foundation_year_timeline(df_plot):
    """Creates a Plotly Line chart for company foundations over time."""
    if 'Founded Year' not in df_plot.columns or df_plot['Founded Year'].isnull().all():
         return None
    yearly_counts = df_plot['Founded Year'].value_counts().sort_index()
    current_year = datetime.datetime.now().year
    yearly_counts = yearly_counts[(yearly_counts.index > 1950) & (yearly_counts.index <= current_year)] # Reasonable range
    if yearly_counts.empty: return None

    fig = px.line(x=yearly_counts.index, y=yearly_counts.values, markers=False, # Smoother look
                  labels={'x': 'Year Founded', 'y': 'Number of Companies Founded'}, title='Company Foundations Over Time')
    fig.add_traces(go.Scatter(x=yearly_counts.index, y=yearly_counts.values, fill='tozeroy',
                              fillcolor='rgba(16, 55, 120, 0.2)', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.update_layout(title_font_size=20, font_size=14, plot_bgcolor='white',
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                      margin=dict(t=50, b=20, l=20, r=20))
    fig.update_traces(line=dict(color='#103778', width=2.5))
    return fig
# --- End Plotting Functions ---


# --- Topic Modeling Functions ---
@st.cache_resource # Cache the fitted model and results
def run_bertopic(texts, n_topics=10):
    """Runs BERTopic modeling."""
    if not texts: return None, None
    try:
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        vectorizer = CountVectorizer(stop_words="english")
        topic_model = BERTopic(embedding_model=sentence_model, vectorizer_model=vectorizer,
                               nr_topics=n_topics, calculate_probabilities=True, verbose=False) # verbose=False for cleaner UI
        topics, _ = topic_model.fit_transform(texts)
        return topic_model, topics
    except Exception as e:
        st.error(f"Error during BERTopic modeling: {e}")
        return None, None

@st.cache_resource
def run_corex(texts, n_topics=10):
    """Runs CorEx topic modeling."""
    if not texts: return None, None, None
    try:
        vectorizer = CountVectorizer(stop_words='english', max_features=5000, token_pattern=r'\b[a-zA-Z]{3,}\b')
        X = vectorizer.fit_transform(texts)
        if X.shape[1] == 0:
            st.error("No valid words found after preprocessing for CorEx.")
            return None, None, None
        words = vectorizer.get_feature_names_out()
        topic_model = ct.Corex(n_hidden=n_topics, seed=42)
        topic_model.fit(X, words=words)
        topics = [topic_model.get_topics(topic=i, n_words=10) for i in range(n_topics)]
        return topic_model, topics, words
    except Exception as e:
        st.error(f"Error during CorEx modeling: {e}")
        return None, None, None
# --- End Topic Modeling Functions ---


# --- RAG Setup and Query Functions ---
@st.cache_resource # Cache the RAG chain
def setup_rag_for_competitor_analysis(_df_rag): # Use _ prefix for cache key based on data
    """Sets up the RAG chain using FAISS and an LLM."""
    global LLM_INITIALIZED # Access global flag

    if not RAG_ENABLED or _df_rag.empty or 'Description' not in _df_rag.columns or _df_rag['Description'].isnull().all():
        st.warning("Cannot set up RAG: Requires API Key and 'Description' data.")
        return None
    try:
        documents = []
        for _, row in _df_rag.iterrows():
            content = f"Company Name: {row.get('Company Name', 'N/A')}\n"
            if pd.notna(row.get('Description')): content += f"Description: {row['Description']}\n"
            for col in ['Topic', 'City', 'Founded Year']:
                if col in _df_rag.columns and pd.notna(row.get(col)):
                    key_name = 'Industry/Topic' if col == 'Topic' else col
                    content += f"{key_name}: {row[col]}\n"
            # Add other non-empty, non-standard columns
            other_cols = [c for c in _df_rag.columns if c not in ['Company Name', 'Description', 'Topic', 'City', 'Founded Year', 'Company Age', 'Age Group'] and pd.notna(row.get(c))]
            for col in other_cols: content += f"{col}: {row[col]}\n"
            documents.append(Document(page_content=content.strip(), metadata={"company": row.get('Company Name', 'N/A')}))

        if not documents: st.error("No documents created for RAG."); return None

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Initialize LLM (prioritize Groq if available and key is set)
        llm = None
        llm_provider = None
        if USE_GROQ and GROQ_API_KEY:
            try:
                llm = ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY, temperature=0.7)
                llm_provider = "Groq"
            except Exception as e: st.warning(f"Groq init failed: {e}. Trying OpenAI.")
        if llm is None and USE_OPENAI and OPENAI_API_KEY:
            try:
                llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.7)
                llm_provider = "OpenAI"
            except Exception as e: st.error(f"OpenAI init failed: {e}"); return None
        if llm is None: st.error("LLM initialization failed. Cannot create RAG chain."); return None

        LLM_INITIALIZED = True # Set flag
        st.sidebar.success(f"RAG System Ready ({llm_provider})")
        return ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 7}), # Retrieve more docs
            return_source_documents=True)

    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        return None

def find_potential_competitors(company_name, company_details, retrieval_chain):
    """Queries the RAG chain to find competitors."""
    if retrieval_chain is None: return "Competitor analysis system not available.", []
    try:
        query = f"""
        Analyze the provided database of Irish CHGFs to find potential competitors for: "{company_name}".
        Company Details: {company_details}

        Identify the top 3-5 most similar companies based on business description, industry/sector, and target market.
        For each potential competitor found *only* in the database:
        1. Provide Company Name, Description, and Industry/Topic.
        2. Briefly explain the key similarity making them a competitor.

        Exclude "{company_name}" itself. Format clearly, starting with a summary.
        If no strong competitors are found, state that clearly.
        """
        result = retrieval_chain({"question": query, "chat_history": []})
        llm_answer = result.get("answer", "Analysis could not be generated.")

        # Extract source documents, parse, and filter
        competitors = []
        seen_companies = {company_name.lower()} # Exclude self
        if "source_documents" in result:
            for doc in result["source_documents"]:
                company_info = {}
                comp_name_found = None
                for line in doc.page_content.strip().split('\n'):
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        key, value = key.strip(), value.strip()
                        company_info[key] = value
                        if key == "Company Name": comp_name_found = value
                # Add if valid, not self, and not already added
                if comp_name_found and comp_name_found.lower() not in seen_companies:
                    if 'Company Name' in company_info:
                        competitors.append(company_info)
                        seen_companies.add(comp_name_found.lower())
                        if len(competitors) >= 5: break # Limit to top 5 distinct competitors found

        return llm_answer, competitors
    except Exception as e:
        st.error(f"Error finding competitors: {str(e)}")
        return "An error occurred during competitor analysis.", []
# --- End RAG Functions ---


# ======== Main Application Logic ========
def main():
    # --- Page Title and Intro ---
    col1_title, col2_title = st.columns([1, 8], gap="small")
    with col1_title:
         st.markdown("""<div style="background-color: white; padding: 5px; border-radius: 5px; text-align: center; height: 50px; display: flex; align-items: center; justify-content: center;">
            <div style="display: flex; height: 30px; width: 45px; border: 1px solid #ccc;">
                <div style="background-color: #169b62; flex: 1;"></div><div style="background-color: white; flex: 1;"></div><div style="background-color: #ff883e; flex: 1;"></div>
            </div></div>""", unsafe_allow_html=True)
    with col2_title:
        st.title("Irish CHGFs Analysis Dashboard")

    # --- Data Input Section (Sidebar) ---
    st.sidebar.markdown("## 📂 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File (Optional, uses default if available)", type=['xlsx', 'xls'])
    df = load_data(uploaded_file) # Load and clean data

    # Stop if data loading failed
    if df.empty:
        st.error("No data loaded. Please upload a valid Excel file or ensure 'ireland_cleaned_CHGF.xlsx' exists.")
        st.stop()

    # --- Initialize RAG System (if not already done) ---
    if RAG_ENABLED and st.session_state.retrieval_chain is None:
        st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)


    # --- Sidebar Filters ---
    st.sidebar.markdown("## 📊 Global Filters")
    df_filtered = df.copy() # Start filtering from the full, cleaned dataframe

    # Topic filter
    if 'Topic' in df_filtered.columns:
        all_topics = sorted([t for t in df_filtered['Topic'].unique() if pd.notna(t) and t != 'Uncategorized'])
        if all_topics:
            selected_topics = st.sidebar.multiselect("Filter by Topic", options=all_topics, default=[])
            if selected_topics: df_filtered = df_filtered[df_filtered['Topic'].isin(selected_topics)]

    # City filter
    if 'City' in df_filtered.columns:
         all_cities = sorted([c for c in df_filtered['City'].unique() if pd.notna(c) and c != 'Unknown'])
         if all_cities:
            selected_cities = st.sidebar.multiselect("Filter by City", options=all_cities, default=[])
            if selected_cities: df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]

    # Company age filter
    if 'Company Age' in df_filtered.columns and not df_filtered['Company Age'].isnull().all():
         min_age, max_age = int(df_filtered['Company Age'].min()), int(df_filtered['Company Age'].max())
         if max_age > min_age: # Only show slider if there's a range
            age_range = st.sidebar.slider("Company Age Range", min_age, max_age, (min_age, max_age))
            df_filtered = df_filtered[df_filtered['Company Age'].between(age_range[0], age_range[1])]

    # Company name search
    company_search = st.sidebar.text_input("Search by Company Name")
    if company_search:
        df_filtered = df_filtered[df_filtered['Company Name'].str.contains(company_search, case=False, na=False)]

    # Filtered data count & download
    st.sidebar.markdown(f"""<div style="background-color: #eef2ff; padding: 10px; border-radius: 5px; margin-top: 15px; text-align: center;">
        <span style="font-weight: 600;">Showing {len(df_filtered)}</span> / {len(df)} companies</div>""", unsafe_allow_html=True)
    if not df_filtered.empty:
        st.sidebar.markdown("<div style='text-align: center; margin-top: 15px;'>"+
                            get_download_link(df_filtered, 'filtered_irish_chgfs.csv', 'Download Filtered Data')+
                            "</div>", unsafe_allow_html=True)
    else:
        st.sidebar.warning("No companies match filters.")


    # --- Main Content Tabs ---
    st.markdown("""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #103778;">
        <p style="margin: 0; font-size: 1.0rem;">Explore data, analyze topics, and find potential competitors among Irish CHGFs.</p>
        </div>""", unsafe_allow_html=True)

    tab_titles = ["📊 Dashboard", "🔍 Company Explorer", "🏷️ Topic Analysis", "🧠 Adv. Topic Modeling", "🥇 Competitor Analysis"]
    tabs = st.tabs(tab_titles)


    # ======== TAB 0: Dashboard ========
    with tabs[0]:
        st.header("Dashboard Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f"<div class='metric-card'><span style='font-size: 1.75rem; font-weight: 600;'>{len(df)}</span><p>Total Companies</p></div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='metric-card'><span style='font-size: 1.75rem; font-weight: 600;'>{df['Topic'].nunique()}</span><p>Unique Topics</p></div>", unsafe_allow_html=True)
        with col3: st.markdown(f"<div class='metric-card'><span style='font-size: 1.75rem; font-weight: 600;'>{df['City'].nunique()}</span><p>Unique Cities</p></div>", unsafe_allow_html=True)
        with col4:
            coverage = round((len(df[df['Topic'] != 'Uncategorized']) / len(df)) * 100, 1) if len(df) > 0 else 0
            st.markdown(f"<div class='metric-card'><span style='font-size: 1.75rem; font-weight: 600;'>{coverage}%</span><p>Topic Coverage</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if df['Topic'].nunique() > 1:
                 fig = create_city_pie_chart(df_filtered if not df_filtered.empty else df) # Show filtered or full
                 if fig: st.plotly_chart(fig, use_container_width=True)
                 else: st.info("Could not generate City Distribution chart.")
            fig = create_company_age_chart(df_filtered if not df_filtered.empty else df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Could not generate Company Age Distribution chart.")
        with chart_col2:
            fig = create_foundation_year_timeline(df_filtered if not df_filtered.empty else df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Could not generate Foundation Year Timeline chart.")

        st.markdown("---")
        st.subheader("Sample Data Preview")
        st.dataframe(df.head(10), use_container_width=True)


    # ======== TAB 1: Company Explorer ========
    with tabs[1]:
        st.header("Company Explorer")
        if df_filtered.empty:
             st.warning("No companies match the current filter criteria set in the sidebar.")
        else:
            col1_exp, col2_exp = st.columns([1, 2])
            with col1_exp:
                st.markdown("#### Display Options")
                sort_options = [col for col in ["Company Name", "Topic", "City", "Founded Year", "Company Age"] if col in df_filtered.columns]
                sort_by = st.selectbox("Sort by", sort_options, key="explorer_sort")
                sort_asc = st.radio("Order", ["Ascending", "Descending"], key="explorer_order") == "Ascending"
                df_display = df_filtered.sort_values(by=sort_by, ascending=sort_asc, na_position='last')
            with col2_exp:
                items_per_page = st.slider("Companies per page", 5, 50, 10, key="explorer_paginate")
                total_pages = max(1, (len(df_display) + items_per_page - 1) // items_per_page)
                current_page = st.number_input("Page", 1, total_pages, 1, 1, key="explorer_page_num")
                start_idx, end_idx = (current_page - 1) * items_per_page, min(current_page * items_per_page, len(df_display))
                st.caption(f"Showing {start_idx + 1} - {end_idx} of {len(df_display)} companies")

            st.markdown("---")
            for _, row in df_display.iloc[start_idx:end_idx].iterrows():
                 with st.expander(f"{row.get('Company Name', 'N/A')}", expanded=False):
                    exp_col1, exp_col2 = st.columns([1, 3]) # Sidebar-like column, main content column
                    with exp_col1:
                        st.markdown(f"**Topic:** {row.get('Topic', 'N/A')}")
                        st.markdown(f"**City:** {row.get('City', 'N/A')}")
                        if pd.notna(row.get('Founded Year')): st.markdown(f"**Founded:** {int(row['Founded Year'])}")
                        if pd.notna(row.get('Company Age')): st.markdown(f"**Age:** {int(row['Company Age'])} years")
                    with exp_col2:
                        st.markdown(f"**Description:**")
                        desc = row.get('Description', 'N/A')
                        st.caption(desc if desc else "No description available.")
                        # Show other non-empty cols concisely
                        other_data = {k:v for k,v in row.items() if k not in ['Company Name','Topic','City','Founded Year','Company Age','Description','Age Group'] and pd.notna(v)}
                        if other_data: st.json(other_data, expanded=False)


    # ======== TAB 2: Topic Analysis ========
    with tabs[2]:
        st.header("Topic Analysis")

        if 'Topic' not in df.columns or 'Description' not in df.columns:
            st.warning("Topic analysis requires both 'Topic' and 'Description' columns.")
        else:
            all_topics = sorted([topic for topic in df['Topic'].unique() if pd.notna(topic)])
            if not all_topics:
                st.warning("No valid topics found in the 'Topic' column.")
            else:
                st.markdown("#### Select Topic to Analyze")
                topic_for_analysis = st.selectbox("", options=all_topics, key="topic_analysis_select_tab2")

                # Defensive Initialization
                combined_description = ""
                topic_companies = pd.DataFrame()

                if topic_for_analysis: # Proceed only if a topic is selected
                    topic_companies = df[df['Topic'] == topic_for_analysis].copy() # Use copy

                    if not topic_companies.empty:
                        combined_description = ' '.join(topic_companies['Description'].fillna('').astype(str))

                    # Display Topic Info / Header
                    st.markdown(f"""<div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #103778;">
                        <h3 style="color: #103778; margin-top: 0;">Analysis: {topic_for_analysis} ({len(topic_companies)} companies)</h3></div>""", unsafe_allow_html=True)

                    # Expander for Companies List
                    with st.expander("View Companies in this Topic", expanded=False):
                        if not topic_companies.empty:
                            st.dataframe(topic_companies[['Company Name', 'Description']], use_container_width=True)
                            st.markdown(get_download_link(topic_companies, f'irish_chgfs_{topic_for_analysis}.csv', f'Download Topic Data'), unsafe_allow_html=True)
                        else: st.write("No companies found for this topic.")

                    # Word Cloud & Common Terms Side-by-Side
                    wc_col, terms_col = st.columns(2)

                    with wc_col:
                        st.subheader("Topic Word Cloud")
                        if combined_description.strip():
                            processed_desc_for_wc = preprocess_text(combined_description)
                            if processed_desc_for_wc:
                                wordcloud_fig = generate_wordcloud(processed_desc_for_wc) # Title added by subheader
                                if wordcloud_fig: st.pyplot(wordcloud_fig)
                            else: st.info("Description empty after preprocessing.")
                        else: st.info("No description data for word cloud.")

                    with terms_col:
                        st.subheader("Most Common Terms")
                        if combined_description.strip():
                            try:
                                stop_words = safe_get_stopwords()
                                tokens = safe_tokenize(preprocess_text(combined_description))
                                words = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2]

                                if words:
                                    word_freq = nltk.FreqDist(words)
                                    top_words = pd.DataFrame(word_freq.most_common(15), columns=['Term', 'Frequency'])
                                    fig_common = px.bar(top_words, x='Frequency', y='Term', orientation='h',
                                                        color='Term', title=f'Top Terms: {topic_for_analysis}', height=450,
                                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                                    fig_common.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False,
                                                             margin=dict(l=10, r=10, t=30, b=10), font_size=12, plot_bgcolor='white')
                                    st.plotly_chart(fig_common, use_container_width=True)
                                else: st.info("No common terms found after filtering.")
                            except Exception as e: st.error(f"Error analyzing terms: {e}")
                        else: st.info("No description data for term analysis.")


    # ======== TAB 3: Advanced Topic Modeling ========
    with tabs[3]:
        st.header("Advanced Topic Modeling")
        if 'Description' not in df.columns:
            st.warning("Advanced Topic Modeling requires the 'Description' column.")
        else:
            descriptions = df['Description'].dropna().astype(str).tolist()
            descriptions = [d for d in descriptions if len(d.split()) > 5] # Basic filter

            if not descriptions:
                st.warning("Not enough valid descriptions found for topic modeling.")
            else:
                st.markdown("<p>Discover underlying themes in company descriptions using BERTopic or CorEx.</p>", unsafe_allow_html=True)
                modeling_method = st.radio("Select Method", ["BERTopic", "CorEx"], key="adv_model_method", horizontal=True)
                num_topics = st.slider("Number of Topics", 5, 30, 10, 1, key="adv_num_topics")

                if st.button(f"Run {modeling_method}", key=f"run_adv_{modeling_method}"):
                     with st.spinner(f"Running {modeling_method}... This may take time."):
                         if modeling_method == "BERTopic":
                             topic_model, topics = run_bertopic(descriptions, num_topics)
                             if topic_model and topics is not None:
                                 st.success("BERTopic modeling complete!")
                                 topic_info = topic_model.get_topic_info()
                                 st.dataframe(topic_info, use_container_width=True)
                                 # Add more BERTopic visualizations if desired
                         elif modeling_method == "CorEx":
                             topic_model, topics, words = run_corex(descriptions, num_topics)
                             if topic_model and topics:
                                 st.success("CorEx modeling complete!")
                                 st.write(f"Total Correlation (TC): {topic_model.tc:.2f}")
                                 for i, topic_w in enumerate(topics):
                                     st.write(f"**Topic {i}:** {', '.join([w for w, s in topic_w[:8]])}...") # Show top terms


    # ======== TAB 4: Competitor Analysis ========
    with tabs[4]:
        st.header("Competitor Analysis Chatbot")
        if not RAG_ENABLED:
            st.error("Competitor Analysis is disabled. Please ensure a valid Groq or OpenAI API key is set as an environment variable (e.g., GROQ_API_KEY or OPENAI_API_KEY).")
        elif st.session_state.retrieval_chain is None:
             st.error("RAG system failed to initialize. Check data and API keys.")
        else:
            # Input Form
            with st.form("competitor_form_tab4"):
                st.markdown("**Enter Your Company Details:**")
                c1f, c2f = st.columns(2)
                with c1f: company_name_input = st.text_input("Your Company Name*", key="ca_comp_name")
                with c2f:
                     industry_options = sorted([t for t in df['Topic'].unique() if t not in ['Uncategorized', 'Unknown']])
                     industry_type_input = st.selectbox("Your Industry/Sector*", options=[""] + industry_options, key="ca_industry")
                company_description_input = st.text_area("Describe your company (products, market, model)*", height=120, key="ca_desc")
                submitted = st.form_submit_button("Find Potential Competitors")

            # Analysis Execution
            if submitted:
                if not all([company_name_input, industry_type_input, company_description_input]):
                    st.warning("Please fill in all required (*) fields.")
                else:
                    with st.spinner("Analyzing..."):
                        full_desc_query = f"Industry: {industry_type_input}. Description: {company_description_input}"
                        analysis_text, comps_found = find_potential_competitors(company_name_input, full_desc_query, st.session_state.retrieval_chain)
                        st.session_state.competitors_found = comps_found # Update session state
                        st.session_state.last_analysis = analysis_text # Store analysis text

            # Display Last Analysis Results
            if 'last_analysis' in st.session_state:
                 st.markdown("---")
                 st.markdown("#### AI Analysis Summary")
                 st.markdown(f"<div class='chat-container bot-message'>{st.session_state.last_analysis}</div>", unsafe_allow_html=True)

                 if st.session_state.competitors_found:
                     st.markdown("#### Top Matching Companies from Database")
                     num_to_display = min(5, len(st.session_state.competitors_found))
                     cols = st.columns(min(3, num_to_display)) # Max 3 columns
                     for i, comp in enumerate(st.session_state.competitors_found[:num_to_display]):
                         with cols[i % len(cols)]:
                             st.markdown(f"""<div class="company-card" style="border-top-color: #103778;">
                                 <h4>{comp.get('Company Name', '?')}</h4>
                                 <p><strong>Industry:</strong> {comp.get('Industry/Topic', 'N/A')}</p>
                                 <p class='description'>{comp.get('Description', 'N/A')}</p>
                                 </div>""", unsafe_allow_html=True)
                     # Download link for displayed competitors
                     comp_df = pd.DataFrame(st.session_state.competitors_found[:num_to_display])
                     st.markdown(get_download_link(comp_df, 'competitors.csv','Download Competitor Details'), unsafe_allow_html=True)


            # Interactive Chat Follow-up
            st.markdown("---")
            st.markdown("#### Ask Follow-up Questions")
            chat_history_display = st.container(height=300) # Scrollable container for chat
            with chat_history_display:
                for msg in st.session_state.chat_history:
                     role_label = "You" if msg["role"] == "user" else "AI"
                     st.markdown(f"**{role_label}:** {msg['content']}")

            with st.form(key='chat_form_followup', clear_on_submit=True):
                 chat_input = st.text_area("Your question:", key="ca_chat_input", height=100)
                 send_button = st.form_submit_button("Send")

            if send_button and chat_input:
                 st.session_state.chat_history.append({"role": "user", "content": chat_input})
                 context_for_llm = "Analysis Context:\n" + st.session_state.get('last_analysis', "No prior analysis.")[:1000] + "\nUser Question:" # Add context
                 formatted_hist = [(msg["content"], resp["content"]) for msg, resp in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2])] # Basic history format

                 with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.retrieval_chain({"question": context_for_llm + chat_input, "chat_history": formatted_hist})
                        response = result.get("answer", "Sorry, I couldn't process that.")
                    except Exception as e: response = f"Error: {e}"
                 st.session_state.chat_history.append({"role": "assistant", "content": response})
                 st.rerun() # Use st.rerun instead of experimental_rerun

            if st.button("Clear Chat", key="ca_clear_chat"):
                st.session_state.chat_history = []
                st.session_state.competitors_found = []
                if 'last_analysis' in st.session_state: del st.session_state['last_analysis']
                st.rerun()


# --- App Entry Point ---
if __name__ == "__main__":
    # Handle potential asyncio event loop issues, common in Streamlit
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    main()