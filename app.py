import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Iris Dataset EDA",
    page_icon="🌸",
    layout="wide"
)

# Load the iris dataset
iris_df = sns.load_dataset('iris')

# Separate numerical columns
numerical_columns = iris_df.select_dtypes(include=['float64', 'int64']).columns

# Title and introduction
st.title("🌸 Iris Dataset Explorer")
st.markdown("""
This interactive app performs Exploratory Data Analysis (EDA) on the famous Iris dataset.
Explore different visualizations and insights about the three species of iris flowers.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Distribution Analysis", "Feature Relationships", "Outlier Detection"])

# Dataset Overview
if page == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Few Rows")
        st.dataframe(iris_df.head())
        
    with col2:
        st.subheader("Dataset Info")
        st.write(f"**Number of Samples:** {len(iris_df)}")
        st.write(f"**Number of Features:** {len(numerical_columns)}")
        st.write(f"**Target Classes:** {', '.join(iris_df['species'].unique())}")
    
    st.subheader("Statistical Summary")
    st.dataframe(iris_df[numerical_columns].describe())
    
    # Correlation Heatmap (only numerical columns)
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(iris_df[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Distribution Analysis
elif page == "Distribution Analysis":
    st.header("📈 Distribution Analysis")
    
    # Feature selection (only numerical features)
    feature = st.selectbox("Select Feature", numerical_columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        st.subheader(f"Histogram of {feature}")
        fig = px.histogram(iris_df, x=feature, color="species", marginal="box")
        st.plotly_chart(fig)
        
    with col2:
        # KDE Plot
        st.subheader(f"Density Plot of {feature}")
        fig, ax = plt.subplots()
        sns.kdeplot(data=iris_df, x=feature, hue="species", fill=True)
        st.pyplot(fig)

# Feature Relationships
elif page == "Feature Relationships":
    st.header("🔄 Feature Relationships")
    
    # Scatter plot matrix using Plotly (numerical columns only)
    st.subheader("Interactive Scatter Plot Matrix")
    fig = px.scatter_matrix(iris_df, 
                          dimensions=numerical_columns,
                          color="species")
    st.plotly_chart(fig)
    
    # Feature correlation analysis
    st.subheader("Select Features to Compare")
    x_feature = st.selectbox("X-axis Feature", numerical_columns, key='x')
    y_feature = st.selectbox("Y-axis Feature", numerical_columns, key='y')
    
    fig = px.scatter(iris_df, 
                    x=x_feature, 
                    y=y_feature, 
                    color="species",
                    title=f"{x_feature} vs {y_feature}")
    st.plotly_chart(fig)

# Outlier Detection
else:
    st.header("📉 Outlier Detection")
    
    # Box plots for numerical features
    st.subheader("Box Plots by Species")
    fig = plt.figure(figsize=(12, 6))
    for i, feature in enumerate(numerical_columns, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=iris_df, x='species', y=feature)
        plt.title(feature)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistical summary of outliers
    st.subheader("Statistical Summary of Potential Outliers")
    for feature in numerical_columns:
        q1 = iris_df[feature].quantile(0.25)
        q3 = iris_df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = iris_df[(iris_df[feature] < lower_bound) | (iris_df[feature] > upper_bound)]
        
        st.write(f"**{feature}**")
        st.write(f"Number of outliers: {len(outliers)}")
        if len(outliers) > 0:
            st.dataframe(outliers)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")