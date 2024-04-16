import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    st.title("Dataset Exploration App")

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

if __name__ == "__main__":
    main()
