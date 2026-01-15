# vector_store.py
from langchain_community.vectorstores import FAISS
# Usamos la versión InferenceAPI que es ligera
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import os
import streamlit as st
import time

# CAMBIO CLAVE: Usamos el modelo más rápido y disponible de HF
# Este modelo casi nunca da error de "Loading"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_dataframe(csv_path="data/retail_indicators.csv") -> pd.DataFrame:
    """
    Carga el CSV a un DataFrame. Cacheado para reuso.
    """
    if not os.path.exists(csv_path):
        # Fallback por si no encuentra el archivo
        return pd.DataFrame(columns=["Category", "Product", "Month/Year", 
                                     "Projected Sales", "Actual Sales", "Actual Purchases"])
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource(show_spinner=False)
def load_and_index_data(csv_path="data/retail_indicators.csv", persist_dir=".faiss_index"):
    """
    Crea o carga un índice FAISS persistente a partir del CSV.
    """
    # 1. Crear directorio si no existe
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    index_path = os.path.join(persist_dir, "index.faiss")
    store_path = os.path.join(persist_dir, "store.pkl")

    # 2. Configurar Embeddings (con manejo de Token explícito)
    api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
    if not api_key:
        # Intento de rescate si os.environ falló (aunque main.py ya debió inyectarlo)
        if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
            api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        else:
            st.error("Falta el Token de Hugging Face. Configura los Secrets.")
            st.stop()

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name=EMBEDDINGS_MODEL
    )

    # 3. Intentar cargar índice existente
    # Borramos la lógica de carga vieja si cambias de modelo para evitar incompatibilidad de dimensiones
    if os.path.exists(index_path) and os.path.exists(store_path):
        try:
            vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            print(f"Índice corrupto o incompatible, regenerando... {e}")
            # Si falla, regeneramos sin detener la app

    # 4. Construir documentos desde el CSV
    df = load_dataframe(csv_path)
    
    # Protección si el CSV está vacío o no se cargó
    if df.empty:
        return FAISS.from_texts(["No data available"], embeddings)

    docs = []
    for i, row in df.iterrows():
        # Construcción segura de strings (evita errores con valores nulos)
        content = (
            f"Category: {row.get('Category', 'N/A')}, Product: {row.get('Product', 'N/A')}, "
            f"Month/Year: {row.get('Month/Year', 'N/A')}, "
            f"Projected Sales: {row.get('Projected Sales', 0)}, "
            f"Actual Sales: {row.get('Actual Sales', 0)}, "
            f"Actual Purchases: {row.get('Actual Purchases', 0)}"
        )
        metadata = {
            "row_id": int(i),
            "category": str(row.get("Category", "")),
            "product": str(row.get("Product", "")),
            "month_year": str(row.get("Month/Year", ""))
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # 5. Generar Embeddings con reintentos (fix para KeyError: 0)
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(persist_dir)
        return vectorstore
    except Exception as e:
        # Si falla (model loading), esperamos un poco y reintentamos una vez
        if "0" in str(e) or "KeyError" in str(e):
            st.warning("El modelo está cargando en Hugging Face... reintentando en 5 segundos.")
            time.sleep(5)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(persist_dir)
            return vectorstore
        else:
            raise e