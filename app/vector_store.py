from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import streamlit as st


@st.cache_resource
def load_and_index_data(csv_path="data/retail_indicators.csv"):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        content = (
            f"Category: {row['Category']}, Product: {row['Product']}, "
            f"Month/Year: {row['Month/Year']}, Projected Sales: {row['Projected Sales']}, "
            f"Actual Sales: {row['Actual Sales']}, Actual Purchases: {row['Actual Purchases']}"
        )
        documents.append(Document(page_content=content))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
