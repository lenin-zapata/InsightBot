# vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import os
import streamlit as st
import time

# Usamos el modelo más rápido y estable
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_dataframe(csv_path="data/retail_indicators.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["Category", "Product", "Month/Year", 
                                     "Projected Sales", "Actual Sales", "Actual Purchases"])
    df = pd.read_csv(csv_path)
    return df

def wait_for_api_warmup(embeddings, max_retries=10):
    """
    Intenta enviar un texto de prueba a la API. 
    Si la API devuelve error (porque está cargando), espera y reintenta.
    """
    placeholder = st.empty()
    
    for i in range(max_retries):
        try:
            # Intentamos vectorizar una palabra simple
            result = embeddings.embed_query("ping")
            
            # Verificamos si el resultado es válido (una lista de floats)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], float):
                placeholder.empty()
                return True # ¡Éxito!
            
            # Si llegamos aquí, devolvió algo raro pero no dio error de código
            raise ValueError(f"Respuesta inesperada: {result}")

        except Exception as e:
            # Detectamos si es el error de carga típico
            err_msg = str(e).lower()
            if "loading" in err_msg or "503" in err_msg or "keyerror" in err_msg:
                placeholder.warning(f"⏳ Despertando a la IA (Intento {i+1}/{max_retries})... espera unos segundos.")
                time.sleep(5) # Espera 5 segundos entre intentos
            else:
                # Si es otro error (ej: autenticación), fallamos rápido
                placeholder.error(f"Error de API: {e}")
                raise e
    
    placeholder.error("❌ La API de Hugging Face está tardando demasiado. Intenta recargar la página.")
    return False

@st.cache_resource(show_spinner=False)
def load_and_index_data(csv_path="data/retail_indicators.csv", persist_dir=".faiss_index"):
    # 1. Crear directorio
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    index_path = os.path.join(persist_dir, "index.faiss")
    store_path = os.path.join(persist_dir, "store.pkl")

    # 2. Configurar API Key
    api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_key and "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    
    if not api_key:
        st.error("Falta el Token. Revisa los Secrets.")
        st.stop()

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name=EMBEDDINGS_MODEL
    )

    # 3. Cargar índice existente si es válido
    if os.path.exists(index_path) and os.path.exists(store_path):
        try:
            vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception:
            pass # Si falla, regeneramos

    # 4. Procesar CSV
    df = load_dataframe(csv_path)
    if df.empty:
        return FAISS.from_texts(["No data"], embeddings)

    docs = []
    for i, row in df.iterrows():
        content = (
            f"Category: {row.get('Category', '')}, Product: {row.get('Product', '')}, "
            f"Month/Year: {row.get('Month/Year', '')}, "
            f"Projected: {row.get('Projected Sales', 0)}, "
            f"Sales: {row.get('Actual Sales', 0)}, Purchases: {row.get('Actual Purchases', 0)}"
        )
        metadata = {
            "row_id": int(i),
            "product": str(row.get("Product", "")),
            "category": str(row.get("Category", ""))
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # 5. CALENTAMIENTO (La parte crítica) 
    # Antes de mandar todos los documentos, verificamos que la API responda.
    if not wait_for_api_warmup(embeddings):
        st.stop()

    # 6. Crear índice con seguridad
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(persist_dir)
        return vectorstore
    except Exception as e:
        st.error(f"Error creando índices: {e}")
        # Retornamos un store vacío temporal para no romper la app completa
        return FAISS.from_texts(["Error loading data"], embeddings)