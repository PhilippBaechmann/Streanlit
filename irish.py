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
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

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

# Make page wider and fix sidebar text
st.markdown("""
<style>
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100% !important;
    }
    /* Make main page text black */
    .main, .block-container, body, p, span, label, div {
        color: #000000 !important;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8d7c !important;
        color: white !important;
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
    
    /* Make the chat input area larger */
    .stTextArea textarea {
        min-height: 100px !important;
    }
</style>
""", unsafe_allow_html=True)

# Download necessary nltk resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #4e8d7c;
        color: white;
    }
    .stSidebar {
        background-color: #eaeaea;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

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
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Function to create downloadable link
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
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

# Function to generate wordcloud
def generate_wordcloud(text, title=None):
    stopwords_set = set(stopwords.words('english'))
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=stopwords_set,
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=15)
    
    return fig

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
    
    return fig

# Function to create company age distribution chart
def create_company_age_chart(df):
    if 'Founded Year' not in df.columns:
        return None
    
    current_year = 2023  # Using a fixed year for the demo
    
    # Create age groups for better visualization
    df['Age'] = current_year - df['Founded Year']
    
    # Create histogram
    fig = px.histogram(
        df,
        x='Age',
        nbins=20,
        title='Company Age Distribution',
        color_discrete_sequence=['#4e8d7c']
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Company Age (Years)',
        yaxis_title='Number of Companies',
        title_font=dict(size=20, color='#103778'),
        font=dict(color='#000000', size=14)
    )
    
    return fig

# Function to setup RAG for competitor analysis
def setup_rag_for_competitor_analysis(df):
    try:
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
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Initialize the LLM (ChatGroq or ChatOpenAI)
        if USE_GROQ:
            try:
                llm = ChatGroq(
                    model="mixtral-8x7b-32768", # Correct model name for Groq
                    api_key=os.environ.get("GROQ_API_KEY", "")
                )
            except Exception as e:
                st.warning(f"Failed to initialize Groq: {str(e)}. Falling back to OpenAI.")
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                )
        else:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )
        
        # Create retrieval chain
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        return retrieval_chain
    
    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        return None

# Function to find potential competitors
def find_potential_competitors(company_name, company_description, retrieval_chain):
    try:
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
    
    except Exception as e:
        st.error(f"Error finding competitors: {str(e)}")
        return "An error occurred while analyzing competitors. Please try again.", []

# Main application
def main():
    # App title and description
    st.title("Irish Consistent High Growth Firms (CHGFs) Analysis - 2023")
    
    st.markdown("""
    This application provides interactive analysis and visualization of Irish companies identified as 
    Consistent High Growth Firms (CHGFs) in 2023. The dashboard includes data exploration tools,
    topic modeling with BERTopic and CorEx, and advanced filtering capabilities.
    """)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("Please upload a valid Excel file containing the Irish CHGFs data.")
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                if 'Topic' in df.columns:
                    df['Topic'] = df['Topic'].fillna('Uncategorized')
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    if not df.empty:
        # Create tabs for the different sections
        tabs = st.tabs([
            "Dashboard",
            "Company Explorer",
            "Topic Analysis",
            "Advanced Topic Modeling",
            "Competitor Analysis"
        ])
        
        # Initialize the retrieval chain if not already done
        if st.session_state.retrieval_chain is None:
            with st.spinner("Setting up competitor analysis system..."):
                st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)
        
        # Sidebar for global filters
        st.sidebar.title("Global Filters")
        
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
        
        # City filter if available
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
        
        # Company name search
        company_search = st.sidebar.text_input("Search by Company Name")
        if company_search:
            df_filtered = df_filtered[df_filtered['Company Name'].str.contains(company_search, case=False, na=False)]
        
        # Show filtered data count
        st.sidebar.write(f"Showing {len(df_filtered)} out of {len(df)} companies")
        
        # Download filtered data
        if not df_filtered.empty:
            st.sidebar.markdown(
                get_download_link(df_filtered, 'filtered_irish_chgfs.csv', 'Download Filtered Data (CSV)'),
                unsafe_allow_html=True
            )
        
        # Dashboard Overview Tab
        with tabs[0]:
            st.header("Dashboard Overview")
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
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
                if 'Topic' in df.columns:
                    companies_with_topic = len(df[df['Topic'] != 'Uncategorized'])
                    topic_coverage = round((companies_with_topic / len(df)) * 100, 1)
                    st.metric("Topic Coverage", f"{topic_coverage}%")
                else:
                    st.metric("Topic Coverage", "N/A")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Topic Distribution Chart
            st.subheader("Topic Distribution")
            if 'Topic' in df.columns:
                topic_counts = df['Topic'].value_counts().reset_index()
                topic_counts.columns = ['Topic', 'Count']
                
                fig = px.bar(
                    topic_counts, 
                    x='Topic', 
                    y='Count',
                    color='Topic',
                    title='Distribution of Companies by Topic',
                    height=500
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Topic column found in the dataset.")
            
            # City Distribution Chart (if available)
            if 'City' in df.columns:
                st.subheader("City Distribution")
                city_chart = create_city_pie_chart(df)
                if city_chart:
                    st.plotly_chart(city_chart, use_container_width=True)
            
            # Company Age Chart (if available)
            if 'Founded Year' in df.columns:
                st.subheader("Company Age Distribution")
                age_chart = create_company_age_chart(df)
                if age_chart:
                    st.plotly_chart(age_chart, use_container_width=True)
            
            # Word Cloud from all descriptions
            st.subheader("Word Cloud from Company Descriptions")
            if 'Description' in df.columns:
                combined_description = ' '.join(df['Description'].fillna('').astype(str))
                if combined_description.strip():
                    wordcloud_fig = generate_wordcloud(combined_description)
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("No description data available for word cloud generation.")
            else:
                st.info("No Description column found in the dataset.")
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
        
        # Company Explorer Tab
        with tabs[1]:
            st.header("Company Explorer")
            
            # Advanced search and filters
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Additional filters can be added here
                st.subheader("Additional Filters")
                
                # Sort options
                sort_options = ["Company Name", "Topic"]
                if 'Founded Year' in df.columns:
                    sort_options.append("Founded Year")
                if 'City' in df.columns:
                    sort_options.append("City")
                    
                sort_by = st.selectbox("Sort by", sort_options)
                sort_order = st.radio("Sort order", ["Ascending", "Descending"])
                
                # Apply sorting
                if sort_order == "Ascending":
                    df_filtered = df_filtered.sort_values(by=sort_by)
                else:
                    df_filtered = df_filtered.sort_values(by=sort_by, ascending=False)
                
            with col2:
                # Display company count after filtering
                st.subheader(f"Displaying {len(df_filtered)} Companies")
                
                # Create pagination
                items_per_page = st.slider("Companies per page", 5, 50, 10)
                total_pages = max(1, (len(df_filtered) + items_per_page - 1) // items_per_page)
                current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                
                # Calculate start and end indices for pagination
                start_idx = (current_page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(df_filtered))
                
                # Display pagination info
                st.write(f"Showing {start_idx + 1} to {end_idx} of {len(df_filtered)} entries")
            
            # Display company cards
            for idx, row in df_filtered.iloc[start_idx:end_idx].iterrows():
                with st.expander(f"{row['Company Name']}", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.write("**Topic:**")
                        if 'Topic' in df.columns:
                            topic = row['Topic'] if not pd.isna(row['Topic']) else "Uncategorized"
                            st.info(topic)
                        else:
                            st.info("N/A")
                        
                        if 'City' in df.columns and not pd.isna(row['City']):
                            st.write("**City:**")
                            st.info(row['City'])
                        
                        if 'Founded Year' in df.columns and not pd.isna(row['Founded Year']):
                            st.write("**Founded Year:**")
                            st.info(str(int(row['Founded Year'])))
                    
                    with col2:
                        st.write("**Description:**")
                        if 'Description' in df.columns:
                            description = row['Description'] if not pd.isna(row['Description']) else "No description available."
                            st.write(description)
                        else:
                            st.write("No description available.")
                    
                    # Display other columns if available
                    other_cols = [col for col in df.columns if col not in ['Company Name', 'Description', 'Topic', 'City', 'Founded Year']]
                    if other_cols:
                        st.write("**Additional Information:**")
                        for col in other_cols:
                            st.write(f"**{col}:** {row[col]}" if not pd.isna(row[col]) else f"**{col}:** N/A")
        
        # Topic Analysis Tab
        with tabs[2]:
            st.header("Topic Analysis")
            
            if 'Topic' in df.columns and 'Description' in df.columns:
                # Topic selection for analysis
                topic_for_analysis = st.selectbox(
                    "Select Topic to Analyze",
                    options=sorted(df['Topic'].unique())
                )
                
                # Get companies with selected topic
                topic_companies = df[df['Topic'] == topic_for_analysis]
                
                # Display topic information
                st.subheader(f"Analysis of Topic: {topic_for_analysis}")
                st.write(f"Number of companies in this topic: {len(topic_companies)}")
                
                # Display companies in this topic
                with st.expander("Companies in this Topic", expanded=False):
                    st.dataframe(topic_companies[['Company Name', 'Description']], use_container_width=True)
                
                # Create word cloud for this topic
                st.subheader("Topic Word Cloud")
                combined_description = ' '.join(topic_companies['Description'].fillna('').astype(str))
                if combined_description.strip():
                    wordcloud_fig = generate_wordcloud(combined_description, f"Word Cloud for {topic_for_analysis}")
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("No description data available for word cloud generation.")
                
                # Common words analysis
                st.subheader("Most Common Terms")
                if combined_description.strip():
                    # Preprocessing for word frequency
                    stop_words = set(stopwords.words('english'))
                    words = nltk.word_tokenize(combined_description.lower())
                    words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
                    
                    # Calculate word frequencies
                    word_freq = nltk.FreqDist(words)
                    
                    # Display as bar chart
                    top_words = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Frequency'])
                    
                    fig = px.bar(
                        top_words,
                        x='Word',
                        y='Frequency',
                        color='Word',
                        title=f'Most Common Terms in Topic: {topic_for_analysis}',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No description data available for term analysis.")
                
                # Topic similarity analysis
                if len(df['Topic'].unique()) > 1:
                    st.subheader("Topic Similarity Analysis")
                    
                    # Function to calculate topic similarity based on descriptions
                    def calculate_topic_similarity(df):
                        topics = sorted(df['Topic'].unique())
                        similarity_matrix = np.zeros((len(topics), len(topics)))
                        
                        # Create a dictionary of words for each topic
                        topic_words = {}
                        for topic in topics:
                            topic_desc = ' '.join(df[df['Topic'] == topic]['Description'].fillna('').astype(str))
                            words = set(nltk.word_tokenize(preprocess_text(topic_desc)))
                            topic_words[topic] = words
                        
                        # Calculate Jaccard similarity between topics
                        for i, topic1 in enumerate(topics):
                            for j, topic2 in enumerate(topics):
                                if i == j:
                                    similarity_matrix[i, j] = 1.0
                                else:
                                    intersection = len(topic_words[topic1].intersection(topic_words[topic2]))
                                    union = len(topic_words[topic1].union(topic_words[topic2]))
                                    similarity_matrix[i, j] = intersection / union if union > 0 else 0
                        
                        return similarity_matrix, topics
                    
                    # Calculate and visualize topic similarity
                    similarity_matrix, topics = calculate_topic_similarity(df)
                    
                    # Create heatmap for topic similarity
                    fig = px.imshow(
                        similarity_matrix,
                        x=topics,
                        y=topics,
                        color_continuous_scale='Viridis',
                        title='Topic Similarity Matrix (Jaccard Similarity)'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("""
                    This heatmap shows the similarity between topics based on the words used in company descriptions.
                    Darker colors indicate higher similarity. The diagonal is always 1.0 (perfect similarity with itself).
                    """)
            else:
                st.warning("Topic analysis requires both 'Topic' and 'Description' columns in the dataset.")
        
        # Advanced Topic Modeling Tab
        with tabs[3]:
            st.header("Advanced Topic Modeling")
            
            if 'Description' in df.columns:
                # Clean and prepare descriptions
                descriptions = df['Description'].fillna('').astype(str).tolist()
                descriptions = [desc for desc in descriptions if desc.strip()]
                
                if descriptions:
                    # Topic modeling method selection
                    modeling_method = st.radio(
                        "Select Topic Modeling Method",
                        ["BERTopic", "CorEx"]
                    )
                    
                    # Number of topics to generate
                    num_topics = st.slider(
                        "Number of Topics to Generate",
                        min_value=5,
                        max_value=30,
                        value=10,
                        step=1
                    )
                    
                    # Run topic modeling
                    if st.button(f"Run {modeling_method} Topic Modeling"):
                        with st.spinner(f"Running {modeling_method} topic modeling. This may take a few minutes..."):
                            if modeling_method == "BERTopic":
                                try:
                                    # Run BERTopic
                                    topic_model, topics = run_bertopic(descriptions, num_topics)
                                    
                                    # Display topic information
                                    st.subheader("Generated Topics")
                                    
                                    # Get topic info
                                    topic_info = topic_model.get_topic_info()
                                    
                                    # Display topic info table
                                    st.write("Topic Information:")
                                    st.dataframe(topic_info, use_container_width=True)
                                    
                                    # Display top terms for each topic
                                    st.subheader("Top Terms by Topic")
                                    for topic_id in topic_info['Topic'][:num_topics]:
                                        if topic_id != -1:  # Skip outlier topic
                                            topic_terms = topic_model.get_topic(topic_id)
                                            topic_name = f"Topic {topic_id}"
                                            
                                            st.write(f"**{topic_name}**:")
                                            terms_df = pd.DataFrame(topic_terms, columns=["Term", "Score"])
                                            st.dataframe(terms_df, use_container_width=True)
                
                                    # Visualize topic sizes
                                    st.subheader("Topic Size Distribution")
                                    topic_sizes = topic_info.iloc[:num_topics][['Topic', 'Count']]
                                    fig = px.bar(
                                        topic_sizes,
                                        x='Topic',
                                        y='Count',
                                        title='Number of Documents per Topic',
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Visualize topic similarity
                                    st.subheader("Topic Similarity Map")
                                    st.write("This visualization requires a local environment to display. Consider downloading the results and visualizing offline.")
                                    
                                    # Assign topics to companies
                                    st.subheader("Assign Topics to Companies")
                                    if st.button("Update Dataset with BERTopic Topics"):
                                        # Create a copy of the dataframe
                                        df_updated = df.copy()
                                        
                                        # Map topics to company descriptions
                                        topic_assignments = {}
                                        for i, desc in enumerate(descriptions):
                                            if i < len(topics):
                                                topic_assignments[desc] = topics[i]
                                        
                                        # Create new column with BERTopic assignments
                                        df_updated['BERTopic_Topic'] = df_updated['Description'].map(
                                            lambda x: topic_assignments.get(x, -1) if not pd.isna(x) else -1
                                        )
                                        
                                        # Display updated dataframe
                                        st.write("Dataset updated with BERTopic topics:")
                                        st.dataframe(df_updated, use_container_width=True)
                                        
                                        # Download updated dataset
                                        st.markdown(
                                            get_download_link(df_updated, 'irish_chgfs_bertopic.csv', 'Download Dataset with BERTopic Results (CSV)'),
                                            unsafe_allow_html=True
                                        )
                                
                                except Exception as e:
                                    st.error(f"Error in BERTopic modeling: {str(e)}")
                                    st.info("Try with fewer topics or check if you have the required packages installed.")
                            
                            else:  # CorEx
                                try:
                                    # Run CorEx
                                    topic_model, topics, words = run_corex(descriptions, num_topics)
                                    
                                    # Display topic information
                                    st.subheader("Generated Topics")
                                    
                                    # Create a DataFrame to display topic words
                                    topic_words_df = pd.DataFrame()
                                    for i, topic_words in enumerate(topics):
                                        topic_df = pd.DataFrame({
                                            'Word': [word for word, _ in topic_words],
                                            'Weight': [weight for _, weight in topic_words]
                                        })
                                        topic_df['Topic'] = f"Topic {i}"
                                        topic_words_df = pd.concat([topic_words_df, topic_df])
                                    
                                    # Display top terms by topic
                                    st.subheader("Top Terms by Topic")
                                    for topic_num in range(num_topics):
                                        topic_data = topic_words_df[topic_words_df['Topic'] == f"Topic {topic_num}"]
                                        st.write(f"**Topic {topic_num}**:")
                                        st.dataframe(topic_data, use_container_width=True)
                                        
                                        # Generate and display word cloud for each topic
                                        topic_text = ' '.join([f"{row['Word']} " * int(row['Weight'] * 100) 
                                                              for _, row in topic_data.iterrows()])
                                        if topic_text.strip():
                                            wordcloud_fig = generate_wordcloud(topic_text, f"Word Cloud for Topic {topic_num}")
                                            st.pyplot(wordcloud_fig)
                                    
                                    # Topic correlation
                                    st.subheader("Topic Correlation")
                                    st.write("Topic Correlation Matrix:")
                                    
                                    # Get topic correlations
                                    topic_corrs = topic_model.tcs.copy()
                                    
                                    # Create correlation heatmap
                                    fig = px.imshow(
                                        topic_corrs,
                                        x=[f"Topic {i}" for i in range(num_topics)],
                                        y=[f"Topic {i}" for i in range(num_topics)],
                                        color_continuous_scale='Viridis',
                                        title='Topic Correlation Matrix'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Assign topics to companies
                                    st.subheader("Assign Topics to Companies")
                                    if st.button("Update Dataset with CorEx Topics"):
                                        try:
                                            # Get document-topic probabilities
                                            doc_topic_probs = topic_model.p_y_given_x
                                            
                                            # Assign each document to the highest probability topic
                                            doc_topics = np.argmax(doc_topic_probs, axis=1)
                                            
                                            # Create a copy of the dataframe
                                            df_updated = df.copy()
                                            
                                            # Add topic assignments
                                            valid_indices = df_updated['Description'].fillna('').astype(str).str.strip() != ''
                                            df_updated.loc[valid_indices, 'CorEx_Topic'] = doc_topics
                                            
                                            # Display updated dataframe
                                            st.write("Dataset updated with CorEx topics:")
                                            st.dataframe(df_updated, use_container_width=True)
                                            
                                            # Download updated dataset
                                            st.markdown(
                                                get_download_link(df_updated, 'irish_chgfs_corex.csv', 'Download Dataset with CorEx Results (CSV)'),
                                                unsafe_allow_html=True
                                            )
                                        except Exception as e:
                                            st.error(f"Error assigning topics: {str(e)}")
                                            st.info("There was an issue updating the dataset with CorEx topics.")
                                
                                except Exception as e:
                                    st.error(f"Error in CorEx modeling: {str(e)}")
                                    st.info("Try with fewer topics or check if you have the required packages installed.")
                else:
                    st.warning("No valid company descriptions found in the dataset.")
            else:
                st.warning("The dataset does not contain a 'Description' column required for topic modeling.")
        
        # Competitor Analysis Tab (New)
        with tabs[4]:
            st.header("Competitor Analysis Chatbot")
            
            # Introduction to the competitor analysis feature
            st.markdown("""
            <div style="background-color: #f0f9ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0369a1;">
                <h3 style="color: #0369a1; margin-top: 0;">Find Potential Competitors</h3>
                <p>This tool uses AI to identify potential competitors for your company from our database of Irish Consistent High Growth Firms (CHGFs).</p>
                <p>Enter your company details below to get started.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if retrieval chain is available
            if st.session_state.retrieval_chain is None:
                st.warning("The competitor analysis system is not available. Please check your API keys and try reloading the page.")
            else:
                # Company input form
                with st.form("competitor_form"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        company_name = st.text_input("Your Company Name")
                    
                    with col2:
                        industry_type = st.selectbox(
                            "Industry/Sector",
                            options=["Technology", "Finance", "Healthcare", "Manufacturing", "Retail", "Services", "Other"]
                        )
                    
                    company_description = st.text_area(
                        "Describe your company's business model, products/services, and target market",
                        height=150,
                        placeholder="E.g., Our company develops AI-powered financial analytics software for small to medium businesses. We focus on predictive cash flow analysis and offer integration with popular accounting platforms."
                    )
                    
                    advanced_options = st.expander("Advanced Options", expanded=False)
                    with advanced_options:
                        similarity_focus = st.multiselect(
                            "Focus areas for competitor matching",
                            options=["Business Model", "Target Market", "Technology", "Products/Services", "Industry Sector"],
                            default=["Business Model", "Industry Sector"]
                        )
                        
                        num_results = st.slider("Number of competitors to display", min_value=3, max_value=10, value=5)
                    
                    submitted = st.form_submit_button("Find Potential Competitors")
                
                # Process the form submission
                if submitted:
                    if not company_name or not company_description:
                        st.error("Please provide both company name and description to find competitors.")
                    else:
                        with st.spinner("Analyzing potential competitors..."):
                            try:
                                # Append industry information to the description
                                full_description = f"{company_description}\nIndustry/Sector: {industry_type}"
                                
                                # Find potential competitors using RAG
                                analysis, competitors = find_potential_competitors(
                                    company_name, 
                                    full_description, 
                                    st.session_state.retrieval_chain
                                )
                                
                                # Store the found competitors
                                st.session_state.competitors_found = competitors
                                
                                # Display the analysis
                                st.markdown("## Competitor Analysis Results")
                                st.markdown(f"<div class='chat-container bot-message'>{analysis}</div>", unsafe_allow_html=True)
                                
                                # Display competitor cards
                                if competitors:
                                    st.markdown("### Top Matching Companies")
                                    
                                    # Create a grid for competitor cards
                                    cols = st.columns(min(3, len(competitors)))
                                    
                                    for i, competitor in enumerate(competitors[:num_results]):
                                        col_idx = i % len(cols)
                                        with cols[col_idx]:
                                            company_name = competitor.get('Company Name', 'Unknown')
                                            description = competitor.get('Description', 'No description available.')
                                            topic = competitor.get('Industry/Topic', 'Uncategorized')
                                            
                                            st.markdown(f"""
                                            <div style="background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-top: 4px solid #4e8d7c;">
                                                <h4 style="color: #2c3e50; margin-top: 0;">{company_name}</h4>
                                                <p style="font-size: 14px; margin-bottom: 10px;"><strong>Industry:</strong> {topic}</p>
                                                <p style="font-size: 14px;">{description}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.info("No specific competitors were identified. Try adjusting your company description to be more specific.")
                                
                                # Option to download competitor report
                                if competitors:
                                    competitors_df = pd.DataFrame(competitors[:num_results])
                                    st.markdown(
                                        get_download_link(competitors_df, 'competitor_analysis.csv', 'Download Competitor Analysis Report'),
                                        unsafe_allow_html=True
                                    )
                            except Exception as e:
                                st.error(f"Error during competitor analysis: {str(e)}")
                                st.info("There was an issue with the competitor analysis. Please try again with different inputs.")
                
                # Interactive chat with the competitor analysis system
                st.markdown("---")
                st.subheader("Interactive Competitor Analysis Chat")
                
                st.markdown("""
                <div style="background-color: #f0f4c3; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #afb42b;">
                    <p style="margin: 0;">Ask follow-up questions about the competitors or get strategic insights about your competitive landscape.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display chat history
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"""<div class="chat-container user-message">
                            <p><strong>You:</strong> {message["content"]}</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="chat-container bot-message">
                            <p><strong>AI:</strong> {message["content"]}</p>
                        </div>""", unsafe_allow_html=True)
                
                # Chat input
                chat_input = st.text_area("Ask a question about competitors or market positioning:", key="chat_input", height=100)
                
                # Chat buttons
                col1, col2 = st.columns([1, 3])
                with col1:
                    send_button = st.button("Send")
                with col2:
                    clear_button = st.button("Clear Chat History")
                
                if send_button and chat_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": chat_input})
                    
                    # Prepare context for the query
                    context = ""
                    if st.session_state.competitors_found:
                        context = "Competitors found:\n"
                        for comp in st.session_state.competitors_found:
                            comp_name = comp.get("Company Name", "Unknown")
                            comp_desc = comp.get("Description", "No description available.")
                            comp_topic = comp.get("Industry/Topic", "Uncategorized")
                            context += f"- {comp_name} ({comp_topic}): {comp_desc}\n"
                    
                    # Prepare the query with competitor context
                    query = f"""
                    Context about identified competitors:
                    {context}
                    
                    Based on the above competitors and Irish CHGFs dataset, please answer this question:
                    {chat_input}
                    """
                    
                    with st.spinner("Processing your question..."):
                        try:
                            # Get historical chat for context
                            chat_history = [(msg["content"], msg["content"]) for i, msg in enumerate(st.session_state.chat_history[:-1]) if i % 2 == 0 and i+1 < len(st.session_state.chat_history)]
                            
                            # Get response
                            result = st.session_state.retrieval_chain({"question": query, "chat_history": chat_history})
                            response = result["answer"]
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
                            # Add error response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": "I'm sorry, I encountered an error processing your question. Please try again."})
                    
                    # Rerun to update the chat display
                    st.experimental_rerun()
                
                if clear_button:
                    st.session_state.chat_history = []
                    st.experimental_rerun()
                
                # Add suggestions for questions
                with st.expander("Suggested Questions", expanded=False):
                    suggestions = [
                        "What are the key strengths of my competitors?",
                        "How can I differentiate my business from these competitors?",
                        "Are there any potential partnership opportunities with these companies?",
                        "What trends are visible in the competitive landscape?",
                        "Which competitor poses the biggest threat and why?",
                        "What market gaps exist that my company could fill?"
                    ]
                    
                    suggestion_cols = st.columns(2)
                    for i, suggestion in enumerate(suggestions):
                        col_idx = i % 2
                        with suggestion_cols[col_idx]:
                            if st.button(suggestion, key=f"suggestion_{suggestion}"):
                                # Set the chat input value
                                st.session_state.chat_input = suggestion
                                # Rerun to update the input field
                                st.experimental_rerun()

if __name__ == "__main__":
    main()