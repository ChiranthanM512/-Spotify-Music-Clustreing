import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Spotify Music Clustering Dashboard",
    page_icon="🎵",
    layout="wide"
)

# Title
st.title("🎵 Spotify Music Clustering Dashboard")
st.markdown("Analyze Spotify tracks using audio features and clustering.")

# Sidebar
st.sidebar.header("⚙️ Dashboard Settings")

clusters = st.sidebar.slider(
    "Select number of clusters",
    min_value=2,
    max_value=8,
    value=4
)

feature_choice = st.sidebar.selectbox(
    "Feature Explorer",
    ['danceability','energy','tempo','loudness','valence']
)

st.sidebar.markdown("---")
st.sidebar.info("Dataset Source: Kaggle\n\nUltimate Spotify Tracks Dataset")

# Load dataset
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")
    df = pd.read_csv(os.path.join(path, "SpotifyFeatures.csv"))
    return df

df = load_data()

# Feature selection
features = ['danceability','energy','tempo','loudness','valence']
data = df[features].dropna()

# Normalize
data_norm = (data - data.min()) / (data.max() - data.min())

# Clustering
data_norm["Cluster"] = pd.qcut(data_norm["energy"], q=clusters, labels=False)

# Metrics
st.markdown("## 📊 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Songs Analyzed", f"{len(data_norm):,}")
col2.metric("Features Used", len(features))
col3.metric("Clusters Created", clusters)

st.markdown("---")

# Charts
st.markdown("## 📈 Cluster Analysis")

col1, col2 = st.columns(2)

with col1:
    cluster_counts = data_norm["Cluster"].value_counts().sort_index()

    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={"x": "Cluster", "y": "Number of Songs"},
        title="Cluster Distribution",
        color=cluster_counts.index
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    cluster_means = data_norm.groupby("Cluster").mean()

    fig = px.line(
        cluster_means,
        title="Feature Comparison Across Clusters",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Feature explorer
st.markdown("## 🎧 Feature Explorer")

fig = px.histogram(
    data_norm,
    x=feature_choice,
    color="Cluster",
    nbins=50,
    title=f"Distribution of {feature_choice}"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data preview
st.markdown("## 📄 Dataset Preview")

st.dataframe(
    data_norm.head(100),
    use_container_width=True
)

st.markdown("---")
st.caption("Spotify Music Clustering Dashboard | Built with Streamlit")
