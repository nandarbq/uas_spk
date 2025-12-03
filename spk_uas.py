import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering - Robust", layout="wide")
st.title("Clustering (robust) — pilih sendiri kolom numerik & input manual")

# --- Load CSV ---
@st.cache_data
def load_csv(path="penguins.csv"):
    return pd.read_csv(path)

# Let user upload or use local file
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (opsional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = load_csv("penguins.csv")
    except Exception as e:
        st.error("Tidak menemukan penguins.csv di folder. Upload file CSV melalui sidebar.")
        st.stop()

st.subheader("Preview dataset (5 baris)")
st.dataframe(df.head())

# --- Show column names to user ---
st.sidebar.subheader("Kolom pada dataset")
cols = list(df.columns)
st.sidebar.write(cols)

# --- Detect numeric columns automatically ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
st.sidebar.subheader("Kolom numerik terdeteksi")
st.sidebar.write(numeric_cols)

if len(numeric_cols) == 0:
    st.error("Tidak ada kolom numerik terdeteksi. Pastikan file CSV memiliki kolom yang bertipe angka.")
    st.stop()

# --- User chooses features for clustering ---
st.sidebar.subheader("Pilih fitur untuk clustering")
feats = st.sidebar.multiselect("Fitur numerik (minimal 2)", numeric_cols, default=numeric_cols[:4] if len(numeric_cols)>=4 else numeric_cols)

if len(feats) < 2:
    st.warning("Pilih minimal 2 fitur numerik agar bisa divisualisasikan.")
    # still allow but stop further
    st.stop()

# --- Optional: preview statistics ---
st.subheader("Statistik fitur terpilih")
st.write(df[feats].describe())

# --- Preprocessing: drop NA rows for selected features ---
data = df[feats].dropna().reset_index(drop=True)

# --- Normalisasi pilihan ---
normalize = st.sidebar.checkbox("Normalisasi (StandardScaler)", value=True)
scaler = StandardScaler() if normalize else None
X = data.values
X_scaled = scaler.fit_transform(X) if scaler else X

# --- K selection & Elbow ---
st.sidebar.subheader("Parameter K-Means")
k = st.sidebar.slider("Jumlah cluster (k)", 2, 8, 3)

if st.sidebar.button("Tampilkan Elbow (1..10)"):
    inertias = []
    for i in range(1, 11):
        km_tmp = KMeans(n_clusters=i, random_state=42, n_init="auto")
        km_tmp.fit(X_scaled)
        inertias.append(km_tmp.inertia_)
    fig_elb, ax_elb = plt.subplots()
    ax_elb.plot(range(1, 11), inertias, marker="o")
    ax_elb.set_xlabel("K")
    ax_elb.set_ylabel("Inertia (WCSS)")
    ax_elb.set_title("Elbow Method")
    st.pyplot(fig_elb)

# --- Train KMeans on selected features ---
km = KMeans(n_clusters=k, random_state=42)
km.fit(X_scaled)
labels = km.predict(X_scaled)
data_out = data.copy()
data_out["cluster"] = labels

st.subheader("Ringkasan hasil clustering")
st.write("Jumlah tiap cluster:")
st.write(data_out["cluster"].value_counts().sort_index())
st.write("Rata-rata fitur per cluster:")
st.write(data_out.groupby("cluster").mean())

# --- Scatter plot of first two selected features ---
st.subheader("Visualisasi (2 fitur pertama yang dipilih)")
xcol, ycol = feats[0], feats[1]
fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(data_out[xcol], data_out[ycol], c=data_out["cluster"], cmap="tab10", alpha=0.7)
ax.set_xlabel(xcol); ax.set_ylabel(ycol)
ax.set_title(f"{xcol} vs {ycol} (warna = cluster)")
plt.colorbar(scatter, ax=ax, label="cluster")
st.pyplot(fig)

# --- Dynamic manual input form based on chosen features ---
st.subheader("Prediksi cluster dari input manual (isikan nilai untuk fitur terpilih)")

# create number inputs dynamically; provide sensible ranges from data
input_vals = []
cols_minmax = {}
for f in feats:
    col_min = float(np.nanmin(df[f])) if df[f].notna().any() else 0.0
    col_max = float(np.nanmax(df[f])) if df[f].notna().any() else 1.0
    col_mean = float(np.nanmean(df[f])) if df[f].notna().any() else 0.0
    cols_minmax[f] = (col_min, col_max, col_mean)

st.markdown("Isi nilai (default = rata-rata kolom). Jika ingin pakai nilai ekstrem, edit range sesuai kebutuhan.")
cols_layout = st.columns(len(feats))
for i, f in enumerate(feats):
    mn, mx, mv = cols_minmax[f]
    # Expand ranges slightly so user can input outside observed range if needed
    pad = (mx - mn) * 0.1 if mx != mn else 1.0
    v = cols_layout[i].number_input(f"{f}", min_value=mn - pad, max_value=mx + pad, value=mv, format="%.3f")
    input_vals.append(v)

if st.button("Prediksi cluster dari input"):
    arr = np.array([input_vals])
    arr_scaled = scaler.transform(arr) if scaler else arr
    pred = km.predict(arr_scaled)[0]
    st.success(f"Input kamu diprediksi masuk ke **Cluster {pred}**")
    # show centroid of that cluster (in original scale)
    centroids_scaled = km.cluster_centers_
    # convert centroids back if scaled
    if scaler:
        centroids = scaler.inverse_transform(centroids_scaled)
    else:
        centroids = centroids_scaled
    centroid_dict = {feat: float(centroids[pred, idx]) for idx, feat in enumerate(feats)}
    st.write("Centroid cluster (rata-rata fitur cluster) :")
    st.json(centroid_dict)

# --- Allow download of clustered data ---
st.subheader("Download hasil clustering")
csv = data_out.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV hasil (fitur + cluster)", data=csv, file_name="clustered_result.csv", mime="text/csv")
