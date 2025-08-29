# vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import os
import streamlit as st

EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource(show_spinner=False)
def load_dataframe(csv_path="data/retail_indicators.csv") -> pd.DataFrame:
    """
    Carga el CSV a un DataFrame. Cacheado para reuso.
    """
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource(show_spinner=False)
def load_and_index_data(csv_path="data/retail_indicators.csv", persist_dir=".faiss_index"):
    """
    Crea o carga un índice FAISS persistente a partir del CSV.
    """
    os.makedirs(persist_dir, exist_ok=True)
    index_path = os.path.join(persist_dir, "index.faiss")
    store_path = os.path.join(persist_dir, "store.pkl")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Recupera si ya existe
    if os.path.exists(index_path) and os.path.exists(store_path):
        vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    # Construye documentos desde el CSV
    df = load_dataframe(csv_path)
    docs = []
    for i, row in df.iterrows():
        content = (
            f"Category: {row['Category']}, Product: {row['Product']}, "
            f"Month/Year: {row['Month/Year']}, Projected Sales: {row['Projected Sales']}, "
            f"Actual Sales: {row['Actual Sales']}, Actual Purchases: {row['Actual Purchases']}"
        )
        metadata = {
            "row_id": int(i),
            "category": row.get("Category"),
            "product": row.get("Product"),
            "month_year": row.get("Month/Year")
        }
        docs.append(Document(page_content=content, metadata=metadata))

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore
