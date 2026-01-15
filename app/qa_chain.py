# qa_chain.py
import os
import streamlit as st
# CAMBIO: Import desde community
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from vector_store import load_and_index_data

CONDENSE_QUESTION_TMPL = "Reescribe la pregunta de seguimiento como una pregunta independiente en {target_lang}:\n\nHistorial: {chat_history}\nPregunta: {question}\nRespuesta:"
QA_TMPL = "Eres un analista. Responde en {target_lang}. Contexto: {context}\nPregunta: {question}\nRespuesta:"

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TMPL)
QA_PROMPT = PromptTemplate.from_template(QA_TMPL)

@st.cache_resource(show_spinner=False)
def build_qa_chain(model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
    vectorstore = load_and_index_data()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # CAMBIO: LLM directo
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=api_token,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    return qa_chain