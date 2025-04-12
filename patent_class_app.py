# Basic topic modelling for Patents

# 
# VM + Load the code on it + Docker Image -> Push on Cloud RUn 
# poetry ?
# 

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords list
stop_words = stopwords.words('english') + ['example', 'comprise', 'comprises', 'comprising', 'method', 'include', 'system', 'device', 'methodology', 
                                           'technology', 'research', 'processing', 'process', 'analysis', 'compute','approach', 
                                           'application', 'implementation', 'computer', 'task', 'use', 
                                           'data', 'result', 'information', 'solution', 'design', 'used', 
                                           'using', 'based', 'according', 'circuit', 'platform', 'node']
lemmatizer = WordNetLemmatizer()

# Basic text cleaning and preprocessing function
def preprocess(text):
    # Step 1: Lowercase text
    text = text.lower()

    # Step 2: Remove numbers
    text = re.sub(r'\d+', '', text)

    # Step 3: Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Step 4: Tokenize the text into words
    tokens = text.split()

    # Step 5: Remove stopwords (before lemmatization) and words with length <= 2
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Step 6: Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Step 7: Remove stopwords after lemmatization (e.g., 'used' becomes 'use', 'system' stays the same)
    tokens = [word for word in tokens if word not in stop_words]

    # Step 8: Return the cleaned text
    return ' '.join(tokens)

# Function to perform topic modeling and create word clouds
def perform_topic_modeling(df, num_topics):
    # Combine Title + Abstract to form the full text
    df['text'] = df['Title'].astype(str) + " " + df['Abstract'].astype(str)
    
    # Apply preprocessing
    df['cleaned'] = df['text'].apply(preprocess)
    
    # Vectorize the cleaned text using CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['cleaned'])  # Document-Term Matrix
    
    # Perform topic modeling using LDA (Latent Dirichlet Allocation)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    # Get the terms from the LDA model for each topic
    feature_names = vectorizer.get_feature_names_out()
    
    # Create the word clouds for each topic
    wordclouds = []
    for topic_idx, topic in enumerate(lda.components_):
        # Create a dictionary of word frequencies for the current topic
        word_freq = {feature_names[i]: topic[i] for i in range(len(feature_names))}
        # Create word cloud for the topic
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        wordclouds.append((topic_idx, wordcloud))
    
    # Assign each patent to a topic
    topic_assignments = lda.transform(dtm)
    df['Topic'] = topic_assignments.argmax(axis=1)
    
    return wordclouds, df

# Streamlit UI
st.title("Topic Modeling & Word Cloud Generation")

# File upload section
uploaded_file = st.file_uploader("Upload your Excel file", type="csv")

if uploaded_file:
    # Read the uploaded file
    df = pd.read_excel(uploaded_file)

    # Check the first few rows to understand the structure
    st.write(df.head())
    
    # Select number of topics for LDA
    num_topics = st.slider("Select number of topics", min_value=2, max_value=10, value=3)
    
    # Perform topic modeling and get word clouds for each topic
    wordclouds, df = perform_topic_modeling(df, num_topics)
    
    # Create a list of topic labels
    topic_labels = [f"Topic {i+1}" for i in range(num_topics)]
    
    # Dropdown to select a topic
    selected_topic = st.selectbox("Select a Topic", topic_labels)
    
    # Map selected topic to the corresponding topic number (starting from 0)
    topic_index = topic_labels.index(selected_topic)
    
    # Display the word cloud for the selected topic
    st.subheader(f"Word Cloud for {selected_topic}")
    st.image(wordclouds[topic_index][1].to_array())
    
    # Filter patents belonging to the selected topic
    patents_in_topic = df[df['Topic'] == topic_index]
    
    # Display the patents in the selected topic
    st.subheader(f"Patents in {selected_topic}")
    st.write(patents_in_topic[['Patent Number', 'Title']])

# End of Code