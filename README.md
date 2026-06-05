# Aprendizaje No Supervisado: Hierarchical Clustering

Este proyecto es una demostración académica y práctica del algoritmo de aprendizaje no supervisado **Hierarchical Clustering** (Agrupamiento Jerárquico), aplicado para clasificar especies de flores utilizando el *Dataset Iris* sin depender de sus etiquetas originales.

## Contenido del Repositorio

*   **`APRENDIZAJE_NO_SUPERVISADO.ipynb`**: Notebook de Jupyter detallado con toda la fundamentación matemática, el preprocesamiento, análisis exploratorio, extracción del método Ward, cálculo de la métrica *Silhouette Score* y la comparación detallada de las ventajas y limitaciones del modelo.
*   **`app.py`**: Interfaz web interactiva construida en **Streamlit**. Permite manipular dinámicamente el número de clusters y el método de enlace para comprender, en tiempo real, el proceso de agrupamiento en el dendrograma y la división geométrica de las especies biológicas mediante gráficos escalables SVG.
*   **`requirements.txt`**: Dependencias para reproducir el proyecto localmente.

## Cómo ejecutar la Aplicación Interactiva Localmente

1.  **Asegúrate de estar en la carpeta del repositorio y tener Python instalado.**
2.  **Crea y activa un entorno virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # En Windows usa: venv\Scripts\activate
    ```
3.  **Instala las dependencias del proyecto:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Inicia el servidor local de Streamlit:**
    ```bash
    streamlit run app.py
    ```
5.  **Abre en tu navegador:** Ve a `http://localhost:8501` para experimentar en vivo.
