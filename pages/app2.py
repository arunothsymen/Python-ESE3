import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK's punkt tokenizer data
nltk.download('punkt')

# Load dataset (selecting only top 100 entries)
@st.cache_data
def load_data():
    df = pd.read_csv('WomensClothingE-CommerceReviews.csv')
    return df.head(100)

# Text preprocessing
def text_preprocessing(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords, punctuation, and special characters, and convert text to lowercase
    stop_words = set(stopwords.words("english"))
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

    # Join tokens back into a single string
    clean_text = " ".join(tokens)

    return clean_text

# Lemmatization
def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(lemmatized_text)

# Text similarity analysis
def text_similarity_analysis(df, division):
    # Filter dataset based on division name
    division_df = df[df["Division Name"] == division]

    # Text preprocessing
    division_df["Cleaned Review Text"] = division_df["Review Text"].apply(text_preprocessing)
    division_df["Cleaned Review Text"] = division_df["Cleaned Review Text"].apply(perform_lemmatization)

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(division_df["Cleaned Review Text"])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix

# Main function
def main():
    st.title("Text Similarity Analysis")

    # Load data
    df = load_data()

    # Sidebar for selecting division
    division_list = df["Division Name"].unique().tolist()
    division = st.sidebar.selectbox("Select Division Name:", division_list)

    # Perform text similarity analysis
    similarity_matrix = text_similarity_analysis(df, division)

    # Display similarity matrix
    st.subheader("Similarity Matrix")
    st.write(pd.DataFrame(similarity_matrix, columns=df[df["Division Name"] == division]["Review Text"], index=df[df["Division Name"] == division]["Review Text"]))

if __name__ == "__main__":
    main()