import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Halaman utama
st.set_page_config(page_title="Clustering K-Means", layout="wide")
st.title("Clustering K-Means — Pilih Fitur & Coba Input Manual")

# Baca dataset
@st.cache_data
def baca_csv(nama_file="penguins.csv"):
    """Fungsi kecil buat baca file CSV. 
       Disimpan cache biar nggak ngulang load tiap refresh."""
    return pd.read_csv(nama_file)

st.sidebar.header("Data")

file_upload = st.sidebar.file_uploader("Upload CSV (opsional)", type=["csv"])

if file_upload is not None:
    data_raw = pd.read_csv(file_upload)
else:
    try:
        data_raw = baca_csv("penguins.csv")
    except:
        st.error("File penguins.csv tidak ditemukan. Coba upload file lewat sidebar.")
        st.stop()

st.subheader("Tampilan Data (5 baris pertama)")
st.dataframe(data_raw.head())

# Tampilkan kolom dataset
st.sidebar.subheader("Daftar kolom")
semua_kolom = list(data_raw.columns)
st.sidebar.write(semua_kolom)

# Cari kolom yg angkanya valid
kolom_numerik = data_raw.select_dtypes(include=[np.number]).columns.tolist()
st.sidebar.subheader("Kolom numerik terdeteksi")
st.sidebar.write(kolom_numerik)

if len(kolom_numerik) == 0:
    st.error("Dataset ini tidak punya kolom numerik.")
    st.stop()

# Pilih fitur clustering
st.sidebar.subheader("Pilih fitur yang mau dipakai")
fitur = st.sidebar.multiselect(
    "Pilih minimal dua kolom",
    kolom_numerik,
    default=kolom_numerik[:4] if len(kolom_numerik) >= 4 else kolom_numerik
)

if len(fitur) < 2:
    st.warning("Minimal pilih 2 fitur supaya bisa dipetakan.")
    st.stop()

st.subheader("Statistik fitur terpilih")
st.write(data_raw[fitur].describe())

# Buang baris yang ada NaN di fitur terpilih
data_bersih = data_raw[fitur].dropna().reset_index(drop=True)

# Normalisasi 
pakai_scaler = st.sidebar.checkbox("Gunakan normalisasi (StandardScaler)", value=True)

if pakai_scaler:
    scaler = StandardScaler()
    data_input = scaler.fit_transform(data_bersih.values)
else:
    scaler = None
    data_input = data_bersih.values

# Pilih jumlah cluster
st.sidebar.subheader("Parameter K-Means")
jumlah_k = st.sidebar.slider("Jumlah cluster", 2, 8, 3)

# Elbow 
if st.sidebar.button("Hitung Elbow Curve (1–10)"):
    inertia_list = []
    for i in range(1, 11):
        model_tmp = KMeans(n_clusters=i, random_state=42, n_init="auto")
        model_tmp.fit(data_input)
        inertia_list.append(model_tmp.inertia_)
    fig_e, ax_e = plt.subplots()
    ax_e.plot(range(1, 11), inertia_list, marker="o")
    ax_e.set_xlabel("K")
    ax_e.set_ylabel("Inertia")
    ax_e.set_title("Elbow Method")
    st.pyplot(fig_e)

# Jalankan K-Means
model = KMeans(n_clusters=jumlah_k, random_state=42)
cluster_hasil = model.fit_predict(data_input)

hasil = data_bersih.copy()
hasil["cluster"] = cluster_hasil

st.subheader("Ringkasan Clustering")
st.write("Jumlah anggota tiap cluster:")
st.write(hasil["cluster"].value_counts().sort_index())
st.write("Rata-rata per fitur dalam cluster:")
st.write(hasil.groupby("cluster").mean())

# Plot (pakai 2 fitur pertama)
x_ft, y_ft = fitur[0], fitur[1]

st.subheader(f"Visualisasi Cluster: {x_ft} vs {y_ft}")

fig, ax = plt.subplots(figsize=(7, 5))
p = ax.scatter(
    hasil[x_ft],
    hasil[y_ft],
    c=hasil["cluster"],
    cmap="tab10",
    alpha=0.7
)
ax.set_xlabel(x_ft)
ax.set_ylabel(y_ft)
ax.set_title("Plot Cluster")
plt.colorbar(p, ax=ax, label="Cluster")
st.pyplot(fig)

# Prediksi manual
st.subheader("Coba Prediksi Cluster Dari Input Manual")

nilai_input = []
rentang = {}

for f in fitur:
    # Cari nilai min, max, dan rata-rata untuk bantuan input
    minv = float(data_raw[f].min())
    maxv = float(data_raw[f].max())
    meanv = float(data_raw[f].mean())
    rentang[f] = (minv, maxv, meanv)

st.markdown("Silakan isi nilai fitur (default = rata-rata).")

kol_layout = st.columns(len(fitur))

for idx, f in enumerate(fitur):
    minv, maxv, meanv = rentang[f]
    # beri sedikit margin agar user bebas isi
    pad = (maxv - minv) * 0.1 if maxv != minv else 1
    val = kol_layout[idx].number_input(
        f,
        min_value=minv - pad,
        max_value=maxv + pad,
        value=meanv,
        format="%.3f"
    )
    nilai_input.append(val)

if st.button("Prediksi"):
    arr = np.array([nilai_input])
    # scaling kembali kalau dipakai
    if scaler:
        arr_scaled = scaler.transform(arr)
    else:
        arr_scaled = arr
    pred = model.predict(arr_scaled)[0]
    st.success(f"Prediksi masuk cluster: **{pred}**")

    # tunjukkan nilai centroid (kembali ke skala asli)
    if scaler:
        centroid_real = scaler.inverse_transform(model.cluster_centers_)
    else:
        centroid_real = model.cluster_centers_

    centroid_map = {
        f: float(centroid_real[pred, i]) for i, f in enumerate(fitur)
    }

    st.write("Rata-rata cluster (centroid):")
    st.json(centroid_map)

# Download hasil
st.subheader("Download Hasil Clustering")
file_csv = hasil.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV Hasil",
    data=file_csv,
    file_name="hasil_cluster.csv",
    mime="text/csv"
)
