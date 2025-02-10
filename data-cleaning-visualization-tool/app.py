import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

def main():
    st.title('Advanced Data Cleaning and Visualization Tool')
    st.markdown("""
    This application helps you clean, process, and visualize your dataset with advanced features like outlier removal,
    feature engineering, and scaling.
    """)

    st.sidebar.title('Upload & Settings')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            display_raw_data(df)
            df = clean_data(df)
            df = feature_engineering(df)
            df = scaling_normalization(df)
            df = split_data(df)
            visualize_data(df)

def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def display_raw_data(df):
    st.subheader('Raw Data')
    st.dataframe(df.head(50))
    st.sidebar.subheader('Dataset Summary')
    st.sidebar.write(df.describe())
    st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

def clean_data(df):
    st.subheader('Data Cleaning')
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = fix_data_types(df)
    df = remove_duplicates(df)
    return df

def handle_missing_values(df):
    option = st.radio("Handle Missing Values:", ["Remove Rows", "Fill with Mean/Median/Mode"])
    if option == "Remove Rows":
        df = df.dropna()
    else:
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
    return df

def remove_outliers(df):
    if st.checkbox("Remove Outliers (Z-score)"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[(np.abs(zscore(df[numeric_cols])) < 3).all(axis=1)]
    return df

def fix_data_types(df):
    if st.checkbox("Fix Incorrect Data Types"):
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    return df

def remove_duplicates(df):
    if st.checkbox("Remove Duplicate Rows"):
        df = df.drop_duplicates()
    return df

def feature_engineering(df):
    st.subheader("Feature Engineering")
    if st.checkbox("Apply Log Transformation"):
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = np.log1p(df[col])
    if st.checkbox("Label Encode Categorical Features"):
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])
    return df

def scaling_normalization(df):
    st.subheader("Scaling & Normalization")
    scaler_option = st.radio("Select Scaling Method:", ["Standardization (Z-score)", "Min-Max Scaling"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if scaler_option == "Standardization (Z-score)":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def split_data(df):
    st.subheader("Train-Test Split")
    test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    st.write("Train Data Sample:", train_df.head())
    st.write("Test Data Sample:", test_df.head())
    return train_df

def visualize_data(df):
    st.subheader('Data Visualization')
    plot_type = st.selectbox('Select Plot Type', ['Scatter Plot', 'Histogram', 'Correlation Heatmap', 'Box Plot'])
    if plot_type == 'Scatter Plot':
        scatter_plot(df)
    elif plot_type == 'Histogram':
        histogram_plot(df)
    elif plot_type == 'Correlation Heatmap':
        correlation_heatmap(df)
    elif plot_type == 'Box Plot':
        box_plot(df)

def scatter_plot(df):
    cols = df.columns.tolist()
    if len(cols) < 2:
        st.error("Dataset must have at least two columns for scatter plot.")
        return
    x_col = st.selectbox('X-axis', cols, key='scatter_x')
    y_col = st.selectbox('Y-axis', cols, key='scatter_y')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    st.pyplot()

def histogram_plot(df):
    cols = df.columns.tolist()
    col = st.selectbox('Column for Histogram', cols, key='hist')
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=20, kde=True)
    st.pyplot()

def correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()

def box_plot(df):
    cols = df.columns.tolist()
    col = st.selectbox('Column for Box Plot', cols, key='box')
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[col])
    st.pyplot()


def footer():
    st.markdown("""
    ---
    **About:** Data Cleaning and Visualization Tool is designed to simplify preprocessing and visualization, ensuring high accuracy in data-driven projects.
    """)


if __name__ == '__main__':
    main()
