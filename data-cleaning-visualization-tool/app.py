import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def main():
    st.title('Data Cleaning and Visualization Tool')
    st.markdown("""
    This application helps you clean your dataset, visualize the data, and gain insights. 
    Upload your dataset and explore the features below.
    """)

    # Sidebar for file upload and settings
    st.sidebar.title('Upload and Settings')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        @st.cache_data
        def load_data():
            try:
                return pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return None

        df = load_data()

        if df is not None:
            # Show raw data
            st.subheader('Raw Data')
            st.dataframe(df.head(50))

            # Display basic dataset statistics
            st.sidebar.subheader('Dataset Summary')
            st.sidebar.write(df.describe())
            st.sidebar.write(f"Number of Rows: {df.shape[0]}")
            st.sidebar.write(f"Number of Columns: {df.shape[1]}")

            # Data cleaning
            st.subheader('Data Cleaning')
            cleaned_data = clean_data(df)
            st.write("Cleaned Data Sample:", cleaned_data.head(50))

            # Allow downloading of cleaned data
            csv = cleaned_data.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Cleaned Data",
                               data=csv,
                               file_name='cleaned_data.csv',
                               mime='text/csv')

            # Visualization section
            st.subheader('Data Visualization')

            plot_type = st.selectbox('Select plot type', ['Scatter Plot', 'Histogram', 'Correlation Heatmap', 'Box Plot'])

            if plot_type == 'Scatter Plot':
                scatter_plot(cleaned_data)
            elif plot_type == 'Histogram':
                histogram_plot(cleaned_data)
            elif plot_type == 'Correlation Heatmap':
                correlation_heatmap(cleaned_data)
            elif plot_type == 'Box Plot':
                box_plot(cleaned_data)

def clean_data(df):
    """
    Cleans the dataset by handling missing values and duplicates.
    """
    st.write("Cleaning Steps:")
    
    # Handle missing values
    missing_values_option = st.radio(
        "How to handle missing values?",
        ("Remove rows with missing values", "Replace with mean/median/mode"))

    if missing_values_option == "Remove rows with missing values":
        df = df.dropna()
        st.write("Rows with missing values have been removed.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        st.write("Missing values in numeric columns replaced with the mean.")

    # Remove duplicates
    if st.checkbox("Remove duplicate rows"):
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        st.write(f"Removed {before - after} duplicate rows.")

    return df

def scatter_plot(df):
    st.write("Scatter Plot:")
    columns = df.columns.tolist()
    if len(columns) < 2:
        st.error("Dataset must contain at least two columns for scatter plot.")
        return
    x_col = st.selectbox('Select X-axis column', columns, key='scatter_x')
    y_col = st.selectbox('Select Y-axis column', columns, key='scatter_y')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    st.pyplot()

def histogram_plot(df):
    st.write("Histogram:")
    columns = df.columns.tolist()
    if not columns:
        st.error("Dataset must contain at least one column for histogram.")
        return
    col = st.selectbox('Select column for histogram', columns, key='histogram')
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=20, kde=True)
    st.pyplot()

def correlation_heatmap(df):
    st.write("Correlation Heatmap:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns for a correlation heatmap.")
        return
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()

def box_plot(df):
    st.write("Box Plot:")
    columns = df.columns.tolist()
    if not columns:
        st.error("Dataset must contain at least one column for box plot.")
        return
    col = st.selectbox('Select column for box plot', columns, key='boxplot')
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[col])
    st.pyplot()

if __name__ == '__main__':
    main()
