import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK's punkt tokenizer data
nltk.download('punkt')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('WomensClothingE-CommerceReviews.csv')
    return df

# Sidebar for filtering
def sidebar(df):
    st.sidebar.header("Filter Data")

    # Filter by Age
    min_age = st.sidebar.slider("Minimum Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].min()))
    max_age = st.sidebar.slider("Maximum Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].max()))
    filtered_df = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)]

    # Filter by Rating
    min_rating = st.sidebar.slider("Minimum Rating", float(df["Rating"].min()), float(df["Rating"].max()), float(df["Rating"].min()))
    max_rating = st.sidebar.slider("Maximum Rating", float(df["Rating"].min()), float(df["Rating"].max()), float(df["Rating"].max()))
    filtered_df = filtered_df[(filtered_df["Rating"] >= min_rating) & (filtered_df["Rating"] <= max_rating)]

    # Filter by Recommended IND
    recommended = st.sidebar.checkbox("Recommended")
    if recommended:
        filtered_df = filtered_df[filtered_df["Recommended IND"] == 1]

    return filtered_df

# Main function
def main():
    st.title("Combined App")

    # Load data
    df = load_data()

    # Sidebar for filtering
    filtered_df = sidebar(df)

    # Display filtered data
    st.subheader("Filtered Data")
    st.write(filtered_df)

    # Visualization
    st.subheader("Visualization")

    # Show Rating Distribution
    if st.checkbox("Show Rating Distribution"):
        sns.histplot(filtered_df["Rating"], kde=True)
        plt.xlabel("Rating")
        plt.ylabel("Count")
        rating_plot = plt.gcf()  # Get the current figure
        st.pyplot(rating_plot)

    # 3D Plot Visualization
    if st.checkbox("Show 3D Plot"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(filtered_df["Age"], filtered_df["Rating"], filtered_df["Positive Feedback Count"])
        ax.set_xlabel("Age")
        ax.set_ylabel("Rating")
        ax.set_zlabel("Positive Feedback Count")
        three_d_plot = plt.gcf()  # Get the current figure
        st.pyplot(three_d_plot)

    # Function to perform image processing operations
    def image_processing(image, techniques):
        img = Image.open(image)

        if "Resize" in techniques:
            width = st.number_input("Enter width:", value=img.width)
            height = st.number_input("Enter height:", value=img.height)
            img = img.resize((int(width), int(height)))

        if "Grayscale" in techniques:
            img = img.convert("L")

        if "Cropping" in techniques:
            left = st.number_input("Enter left coordinate:", value=0)
            top = st.number_input("Enter top coordinate:", value=0)
            right = st.number_input("Enter right coordinate:", value=img.width)
            bottom = st.number_input("Enter bottom coordinate:", value=img.height)
            img = img.crop((int(left), int(top), int(right), int(bottom)))

        if "Rotation" in techniques:
            angle = st.slider("Enter rotation angle:", min_value=0, max_value=360, value=0)
            img = img.rotate(angle)

        return img

    st.title("Image Processing App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
        techniques = st.multiselect("Select image processing techniques:", ["Resize", "Grayscale", "Cropping", "Rotation"])
        processed_img = image_processing(uploaded_file, techniques)
        st.subheader("Processed Image")
        st.image(processed_img, caption="Processed Image", use_column_width=True)

    # Text preprocessing
    def text_preprocessing(text):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
        clean_text = " ".join(tokens)
        return clean_text

    # Lemmatization
    def perform_lemmatization(text):
        lemmatizer = WordNetLemmatizer()
        lemmatized_text = [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(lemmatized_text)

    # Text similarity analysis
    def text_similarity_analysis(df, division):
        division_df = df[df["Division Name"] == division]
        division_df["Cleaned Review Text"] = division_df["Review Text"].apply(text_preprocessing)
        division_df["Cleaned Review Text"] = division_df["Cleaned Review Text"].apply(perform_lemmatization)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(division_df["Cleaned Review Text"])
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return similarity_matrix

    st.title("Text Similarity Analysis")

    division_list = df["Division Name"].unique().tolist()
    division = st.sidebar.selectbox("Select Division Name:", division_list)

    similarity_matrix = text_similarity_analysis(df, division)

    st.subheader("Similarity Matrix")
    st.write(pd.DataFrame(similarity_matrix, columns=df[df["Division Name"] == division]["Review Text"], index=df[df["Division Name"] == division]["Review Text"]))

if __name__ == "__main__":
    main()
