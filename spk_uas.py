import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering Penguins", layout="wide")
st.title("Clustering Dataset Penguins (K-Means)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("penguins.csv")

df_raw = load_data()

# Bersihkan data (hapus NaN)
df = df_raw.dropna().reset_index(drop=True)

# Fitur numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.header("Pilih Fitur")
feats = st.sidebar.multiselect("Fitur numerik", numeric_cols, default=numeric_cols)

if len(feats) == 0:
    st.warning("Pilih minimal 1 fitur numerik.")
    st.stop()

# Normalisasi
normalize = st.sidebar.checkbox("Normalisasi", value=True)

data = df[feats].copy()

if normalize:
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
else:
    data_scaled = data.values

# Pilih K
st.sidebar.header("K-Means")
k = st.sidebar.slider("Jumlah Cluster (K)", 2, 8, 3)

# Elbow Method
if st.sidebar.button("Elbow Method"):
    inertias = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init="auto")
        km.fit(data_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertias, marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

# K-Means
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = km.fit_predict(data_scaled)

df_result = df.copy()
df_result["cluster"] = labels

# Output
st.subheader("Ringkasan Cluster")
st.write(df_result.groupby("cluster")[feats].mean())

st.write("Jumlah data tiap cluster:")
st.write(df_result["cluster"].value_counts())

# Scatter Plot
if len(feats) >= 2:
    x, y = feats[0], feats[1]
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    for c in sorted(df_result["cluster"].unique()):
        sub = df_result[df_result["cluster"] == c]
        ax2.scatter(sub[x], sub[y], label=f"Cluster {c}", alpha=0.7)

    ax2.set_xlabel(x)
    ax2.set_ylabel(y)
    ax2.set_title(f"{x} vs {y}")
    ax2.legend()
    st.pyplot(fig2)

# Download
csv = df_result.to_csv(index=False).encode("utf-8")
st.download_button("Download Hasil", csv, "penguins_clustered.csv", "text/csv")
