from langchain_community.vectorstores import FAISS
# CAMBIO CRÍTICO: Usamos la versión de API, no la local
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import os
import streamlit as st

# Usamos un modelo ligero y multilingüe
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource(show_spinner=False)
def load_dataframe(csv_path="data/retail_indicators.csv") -> pd.DataFrame:
    """
    Carga el CSV a un DataFrame. Cacheado para reuso.
    """
    if not os.path.exists(csv_path):
        st.error(f"No se encontró el archivo {csv_path}. Verifica la carpeta 'data'.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource(show_spinner=False)
def load_and_index_data(csv_path="data/retail_indicators.csv", persist_dir=".faiss_index"):
    """
    Crea o carga un índice FAISS persistente usando la API de Hugging Face para embeddings.
    """
    os.makedirs(persist_dir, exist_ok=True)
    index_path = os.path.join(persist_dir, "index.faiss")
    store_path = os.path.join(persist_dir, "store.pkl")

    # Obtenemos el token del entorno (ya inyectado en main.py)
    api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
    if not api_key:
        raise ValueError("Falta el HUGGINGFACEHUB_API_TOKEN para generar los embeddings.")

    # Inicializamos embeddings vía API (No descarga nada pesado)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name=EMBEDDINGS_MODEL
    )

    # Recupera si ya existe el índice
    if os.path.exists(index_path) and os.path.exists(store_path):
        try:
            vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            st.warning(f"Índice antiguo incompatible o corrupto, regenerando... Error: {e}")

    # Construye documentos desde el CSV si no existe el índice
    df = load_dataframe(csv_path)
    if df.empty:
        # Retorna un vectorstore vacío temporal si falla el CSV para no romper la app
        return FAISS.from_texts(["No data"], embeddings)

    docs = []
    for i, row in df.iterrows():
        content = (
            f"Category: {row.get('Category', '')}, Product: {row.get('Product', '')}, "
            f"Month/Year: {row.get('Month/Year', '')}, Projected Sales: {row.get('Projected Sales', 0)}, "
            f"Actual Sales: {row.get('Actual Sales', 0)}, Actual Purchases: {row.get('Actual Purchases', 0)}"
        )
        metadata = {
            "row_id": int(i),
            "category": row.get("Category"),
            "product": row.get("Product"),
            "month_year": row.get("Month/Year")
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # Creamos el índice (esto llama a la API para vectorizar)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore