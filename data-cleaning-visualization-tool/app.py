import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def main():
    st.title('Data Cleaning and Visualization Tool')

    # File upload and data cleaning section
    st.sidebar.title('Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        @st.cache_data
        def load_data():
            return pd.read_csv(uploaded_file)

        df = load_data()

        # Show raw data
        st.subheader('Raw Data')
        st.write(df)

        # Data cleaning
        st.subheader('Cleaned Data')
        cleaned_data = clean_data(df)
        st.write(cleaned_data)

        # Visualization section
        st.subheader('Data Visualization')

        # Example plot
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plot_type = st.selectbox('Select plot type', ['Scatter Plot', 'Histogram'])

        if plot_type == 'Scatter Plot':
            scatter_plot(cleaned_data)
        elif plot_type == 'Histogram':
            histogram_plot(cleaned_data)

def clean_data(df):
    # Example cleaning function (replace with your cleaning logic)
    cleaned_df = df.dropna()
    return cleaned_df

def scatter_plot(df):
    st.write("Columns available in the dataset:", df.columns)
    columns = df.columns.tolist()
    if len(columns) < 2:
        st.error("Dataset must contain at least two columns for scatter plot.")
        return
    x_col = st.selectbox('Select X-axis column', columns)
    y_col = st.selectbox('Select Y-axis column', columns)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    st.pyplot()

def histogram_plot(df):
    # Example histogram plot (replace with your visualization logic)
    st.write("Columns available in the dataset:", df.columns)
    columns = df.columns.tolist()
    if not columns:
        st.error("Dataset must contain at least one column for histogram.")
        return
    col = st.selectbox('Select column for histogram', columns)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=20, kde=True)
    st.pyplot()

if __name__ == '__main__':
    main()
