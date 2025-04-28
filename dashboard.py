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
from langchain.vectorstores import FAISS
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
    .stTextInput input, .stTextArea textarea, .stSelectbox, .stMultiSelect {
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

# Download necessary nltk resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None

if 'competitors_found' not in st.session_state:
    st.session_state.competitors_found = []

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('ireland_cleaned_CHGF.xlsx')
        # Fill missing Topic values with "Uncategorized"
        if 'Topic' in df.columns:
            df['Topic'] = df['Topic'].fillna('Uncategorized')
        # Add current year for company age calculation if "Founded Year" exists
        if 'Founded Year' in df.columns:
            current_year = datetime.datetime.now().year
            df['Company Age'] = current_year - df['Founded Year']
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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
    # Initialize sentence transformer model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize BERTopic
    vectorizer = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        embedding_model=sentence_model,
        vectorizer_model=vectorizer,
        nr_topics=n_topics,
        verbose=True
    )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(texts)
    
    return topic_model, topics

# Topic modeling with CorEx
@st.cache_resource
def run_corex(texts, n_topics=10):
    # Preprocess texts
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    
    # Train CorEx topic model
    topic_model = ct.Corex(n_hidden=n_topics)
    topic_model.fit(X, words=words)
    
    # Get topics
    topics = [topic_model.get_topics(topic=i, n_words=10) for i in range(n_topics)]
    
    return topic_model, topics, words

# Function to preprocess text
@st.cache_data
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Enhanced wordcloud function
def generate_wordcloud(text, title=None):
    stopwords_set = set(stopwords.words('english'))
    wordcloud = WordCloud(
        width=1000, 
        height=500, 
        background_color='white',
        colormap='viridis',
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

# Function to setup RAG for competitor analysis
def setup_rag_for_competitor_analysis(df):
    # Create a list of documents from the dataframe
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
        
        # Add any other relevant columns
        other_cols = [col for col in df.columns if col not in ['Company Name', 'Description', 'Topic', 'City', 'Founded Year']]
        for col in other_cols:
            if not pd.isna(row[col]):
                content += f"{col}: {row[col]}\n"
        
        doc = Document(page_content=content, metadata={"source": idx, "company": row['Company Name']})
        documents.append(doc)
    
    # Initialize embeddings
    embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed_func(texts):
        return embeddings.encode(texts).tolist()
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embed_func)
    
    # Initialize the LLM (ChatGroq or ChatOpenAI)
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
    
    # Create retrieval chain
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    
    return retrieval_chain

# Function to find potential competitors
def find_potential_competitors(company_name, company_description, retrieval_chain):
    query = f"""
    Find potential competitors for a company named "{company_name}" with the following description:
    {company_description}
    
    Analyze the similarities in business models, industry/sector, and target markets.
    Provide a detailed analysis of the top 3-5 potential competitors.
    """
    
    result = retrieval_chain({"question": query, "chat_history": []})
    
    # Extract source documents (competitors)
    competitors = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            company_info = {}
            lines = doc.page_content.strip().split('\n')
            for line in lines:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    company_info[key] = value
            
            if 'Company Name' in company_info:
                competitors.append(company_info)
    
    return result["answer"], competitors

# Function to create city distribution pie chart
def create_city_pie_chart(df):
    if 'City' not in df.columns:
        return None
    
    # Count city occurrences
    city_counts = df['City'].value_counts()
    
    # For better visualization, limit to top 10 cities and group the rest as "Others"
    if len(city_counts) > 10:
        top_cities = city_counts.head(10)
        others_count = city_counts[10:].sum()
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
    if 'Founded Year' not in df.columns:
        return None
    
    current_year = datetime.datetime.now().year
    
    # Create age groups for better visualization
    df['Age Group'] = pd.cut(
        current_year - df['Founded Year'], 
        bins=[0, 3, 5, 10, 15, 20, 100],
        labels=['0-3 years', '3-5 years', '5-10 years', '10-15 years', '15-20 years', '20+ years']
    )
    
    age_distribution = df['Age Group'].value_counts().sort_index()
    
    # Create bar chart
    fig = px.bar(
        x=age_distribution.index,
        y=age_distribution.values,
        color=age_distribution.index,
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
    if 'Founded Year' not in df.columns:
        return None
    
    # Group by foundation year
    yearly_counts = df['Founded Year'].value_counts().sort_index()
    
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
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    fig.update_traces(
        line=dict(color='#103778', width=3),
        marker=dict(size=8, color='#103778')
    )
    
    return fig

# Main application
def main():
    # App title and header with logo
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Create a simplified Irish flag icon
        irish_flag = """
        <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <div style="display: flex; height: 60px;">
                <div style="background-color: #169b62; flex: 1;"></div>
                <div style="background-color: white; flex: 1;"></div>
                <div style="background-color: #ff883e; flex: 1;"></div>
            </div>
        </div>
        """
        st.markdown(irish_flag, unsafe_allow_html=True)
    
    with col2:
        st.title("Irish Consistent High Growth Firms (CHGFs) Analysis - 2023")
    
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #103778;">
        <p style="margin: 0; font-size: 16px;">
        This application provides interactive analysis and visualization of Irish companies identified as 
        <strong>Consistent High Growth Firms (CHGFs)</strong> in 2023. The dashboard includes data exploration tools,
        topic modeling with <strong>BERTopic</strong> and <strong>CorEx</strong>, and advanced filtering capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("Please upload a valid Excel file containing the Irish CHGFs data.")
        
        # Styled file uploader
        st.markdown("""
        <div style="background-color: #f0f9ff; padding: 20px; border-radius: 8px; border: 1px dashed #3b82f6;">
            <h3 style="color: #1e40af; margin-top: 0;">Upload Dataset</h3>
            <p>Upload your Excel file with Irish CHGFs data to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            if 'Topic' in df.columns:
                df['Topic'] = df['Topic'].fillna('Uncategorized')
    
    if not df.empty:
        # Create tabs for the different sections
        tabs = st.tabs([
            "📊 Dashboard",
            "🔍 Company Explorer",
            "🏷️ Topic Analysis",
            "🧠 Advanced Topic Modeling",
            "🥇 Competitor Analysis"
        ])
        
        # Initialize the retrieval chain if not already done
        if st.session_state.retrieval_chain is None:
            with st.spinner("Setting up competitor analysis system..."):
                st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)
        
        # Sidebar for global filters
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #103778; margin-bottom: 5px;">Global Filters</h2>
            <div style="height: 2px; background-color: #103778; margin-bottom: 20px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display all available topics for filtering
        if 'Topic' in df.columns:
            all_topics = sorted(df['Topic'].unique())
            selected_topics = st.sidebar.multiselect(
                "Filter by Topic", 
                options=all_topics,
                default=[]
            )
            
            # Apply topic filter if selected
            if selected_topics:
                df_filtered = df[df['Topic'].isin(selected_topics)]
            else:
                df_filtered = df
        else:
            df_filtered = df
            st.sidebar.warning("No Topic column found in the dataset.")
        
        # City filter
        if 'City' in df.columns:
            all_cities = sorted(df['City'].unique())
            selected_cities = st.sidebar.multiselect(
                "Filter by City", 
                options=all_cities,
                default=[]
            )
            
            # Apply city filter if selected
            if selected_cities:
                df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]
        
        # Company age filter (if Founded Year exists)
        if 'Founded Year' in df.columns:
            current_year = datetime.datetime.now().year
            min_age = 0
            max_age = int(current_year - df['Founded Year'].min())
            
            age_range = st.sidebar.slider(
                "Company Age Range (years)",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age)
            )
            
            # Apply age filter
            min_year = current_year - age_range[1]
            max_year = current_year - age_range[0]
            df_filtered = df_filtered[(df_filtered['Founded Year'] >= min_year) & (df_filtered['Founded Year'] <= max_year)]
        
        # Company name search
        company_search = st.sidebar.text_input("Search by Company Name")
        if company_search:
            df_filtered = df_filtered[df_filtered['Company Name'].str.contains(company_search, case=False, na=False)]
        
        # Show filtered data count with style
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
        
        # Dashboard Overview Tab
        with tabs[0]:
            st.header("Dashboard Overview")
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Companies", f"{len(df)}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                if 'Topic' in df.columns:
                    st.metric("Number of Topics", f"{len(df['Topic'].unique())}")
                else:
                    st.metric("Number of Topics", "N/A")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                if 'City' in df.columns:
                    st.metric("Cities Represented", f"{len(df['City'].unique())}")
                else:
                    st.metric("Cities Represented", "N/A")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                if 'Topic' in df.columns:
                    companies_with_topic = len(df[df['Topic'] != 'Uncategorized'])
                    topic_coverage = round((companies_with_topic / len(df)) * 100, 1)
                    st.metric("Topic Coverage", f"{topic_coverage}%")
                else:
                    st.metric("Topic Coverage", "N/A")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)