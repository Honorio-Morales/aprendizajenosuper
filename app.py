import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Custom Page Configuration
st.set_page_config(page_title="Hierarchical Clustering Explorer", page_icon=":material/local_florist:", layout="wide")

# Inject Custom CSS for premium aesthetic
st.markdown("""
<style>
    /* Styling adjustments for the app */
    h1 {
        color: #4B0082;
        padding-bottom: 20px;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.title(":material/local_florist: Clustering Jerárquico - Caso Iris")
st.markdown("Explora cómo funciona el algoritmo de **Hierarchical Clustering** de manera interactiva, cambiando los hiperparámetros y observando los resultados en tiempo real.")

# Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    # Preprocessing (excluding the target column)
    X = df.drop('species', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled

df, X_scaled = load_and_preprocess_data()

# ----------------- SIDEBAR -----------------
st.sidebar.header(":material/settings: Parámetros del Algoritmo")
n_clusters = st.sidebar.slider("Número de Clusters (k)", min_value=2, max_value=6, value=3, step=1)
linkage_method = st.sidebar.selectbox("Método de Enlace (Linkage)", options=['ward', 'complete', 'average', 'single'], index=0)

st.sidebar.markdown("""
---
**:material/menu_book: Guía de Enlaces:**
- **Ward**: Minimiza la varianza dentro de los clusters (Esféricos).
- **Complete**: Une basado en la máxima distancia entre puntos de ambos clusters.
- **Average**: Une basado en el promedio de todas las distancias.
- **Single**: Une basado en la mínima distancia (Cadenas).
""")

# ----------------- MAIN PANEL -----------------
col_data, col_dendro = st.columns([1.5, 2.5])

with col_data:
    st.subheader(":material/table: Datos (Resumen)")
    st.dataframe(df.head(8), use_container_width=True)
    st.caption("Solo se visualizan las primeras observaciones.")
    st.info("La columna `species` es puramente referencial; el modelo NO la utiliza para clasificar.")

with col_dendro:
    st.subheader(f":material/account_tree: Dendrograma (Enlace: {linkage_method.capitalize()})")
    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 4.5))
    linked = linkage(X_scaled, method=linkage_method)
    
    # Calculate appropriate distance threshold for the color line
    max_d = linked[-n_clusters+1, 2] if n_clusters > 1 and len(linked) >= n_clusters-1 else 0
    
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False, no_labels=True, ax=ax_dendro, color_threshold=max_d)
    ax_dendro.axhline(y=max_d, color='r', linestyle='--')
    ax_dendro.set_ylabel("Distancia")
    ax_dendro.set_xlabel("Muestras")
    st.pyplot(fig_dendro)

st.markdown("---")
st.subheader(":material/travel_explore: Resultados del Agrupamiento")

# Run Model
try:
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    clusters = hc.fit_predict(X_scaled)
    df['cluster'] = clusters

    # Evaluation Metric
    score = silhouette_score(X_scaled, clusters)
    
    # Metric Display
    col_metric, _ = st.columns([1, 4])
    with col_metric:
        st.metric(label="Silhouette Score (Cohesión)", value=f"{score:.3f}")

    # Visualizations Output
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("##### :material/push_pin: Agrupación Empírica del Modelo")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='cluster', data=df, palette='viridis', s=100, alpha=0.8, legend='full', ax=ax1)
        ax1.set_title(f'{n_clusters} Clusters Encontrados')
        ax1.set_xlabel('Longitud del Pétalo (cm)')
        ax1.set_ylabel('Ancho del Pétalo (cm)')
        st.pyplot(fig1)

    with col_v2:
        st.markdown("##### :material/spa: Especies Reales (Referencia Biológica)")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=df, palette='viridis', s=100, alpha=0.8, legend='full', ax=ax2)
        ax2.set_title('Las 3 Especies Naturales')
        ax2.set_xlabel('Longitud del Pétalo (cm)')
        ax2.set_ylabel('Ancho del Pétalo (cm)')
        st.pyplot(fig2)

except Exception as e:
    st.error(f"Error al calcular los clusters con esos parámetros. Posiblemente los datos formen grupos no manejables con este método. Detalle: {e}")
