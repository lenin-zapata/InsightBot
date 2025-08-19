from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from vector_store import load_and_index_data
import streamlit as st

@st.cache_resource
def build_qa_chain(model_name="flan-t5-small"):
    vectorstore = load_and_index_data()
    retriever = vectorstore.as_retriever()

    pipe = pipeline("text2text-generation", model=f"google/{model_name}")
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

