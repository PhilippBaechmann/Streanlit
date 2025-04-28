#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
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
    /* Make sidebar text white */
    [data-testid="stSidebar"] {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Download necessary nltk resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()



# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
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

# App title and description
def main():
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
            df = pd.read_excel(uploaded_file)
            if 'Topic' in df.columns:
                df['Topic'] = df['Topic'].fillna('Uncategorized')
    
    if not df.empty:
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        pages = ["Dashboard Overview", "Company Explorer", "Topic Analysis", "Advanced Topic Modeling"]
        selected_page = st.sidebar.radio("Go to", pages)
        
        # Sidebar filters (applied globally)
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
        
        # Dashboard Overview Page
        if selected_page == "Dashboard Overview":
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
        
        # Company Explorer Page
        elif selected_page == "Company Explorer":
            st.header("Company Explorer")
            
            # Advanced search and filters
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Additional filters can be added here
                st.subheader("Additional Filters")
                
                # Sort options
                sort_options = ["Company Name", "Topic"]
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
                    
                    with col2:
                        st.write("**Description:**")
                        if 'Description' in df.columns:
                            description = row['Description'] if not pd.isna(row['Description']) else "No description available."
                            st.write(description)
                        else:
                            st.write("No description available.")
                    
                    # Display other columns if available
                    other_cols = [col for col in df.columns if col not in ['Company Name', 'Description', 'Topic']]
                    if other_cols:
                        st.write("**Additional Information:**")
                        for col in other_cols:
                            st.write(f"**{col}:** {row[col]}" if not pd.isna(row[col]) else f"**{col}:** N/A")
        
        # Topic Analysis Page
        elif selected_page == "Topic Analysis":
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
        
        # Advanced Topic Modeling Page
        elif selected_page == "Advanced Topic Modeling":
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
                                    st.error(f"Error in CorEx modeling: {str(e)}")
                                    st.info("Try with fewer topics or check if you have the required packages installed.")
                else:
                    st.warning("No valid company descriptions found in the dataset.")
            else:
                st.warning("The dataset does not contain a 'Description' column required for topic modeling.")

if __name__ == "__main__":
    main()

