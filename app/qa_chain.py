import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from vector_store import load_and_index_data

# -----------------------------
# Prompts
# -----------------------------

CONDENSE_QUESTION_TMPL = """
Eres un asistente que reescribe la siguiente pregunta de seguimiento como una pregunta autónoma y clara en {target_lang}.
Incluye detalles útiles del historial si aplican.

Historial del chat:
{chat_history}

Pregunta de seguimiento:
{question}

Pregunta autónoma en {target_lang}:
""".strip()

QA_TMPL = """
Eres un asistente de analítica retail. Responde en {target_lang}.
Usa EXCLUSIVAMENTE el contexto proporcionado. Si no hay suficiente información,
di que no lo sabes y sugiere qué dato faltaría.

Historial del chat:
{chat_history}

Contexto:
{context}

Pregunta:
{question}

Respuesta en {target_lang}:
""".strip()

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TMPL)
QA_PROMPT = PromptTemplate.from_template(QA_TMPL)

# -----------------------------
# Cadena RAG conversacional
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_qa_chain(model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
    """
    Construye una ConversationalRetrievalChain respaldada por un modelo chat/instruct
    servido por Hugging Face Inference API.
    """
    # Retriever (FAISS + API Embeddings)
    vectorstore = load_and_index_data()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Token explícito
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # LLM (chat) desde HF Inference API
    hf_raw = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
        timeout=120,
        huggingfacehub_api_token=api_token,
    )
    llm = ChatHuggingFace(llm=hf_raw)

    # ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False
    )
    return qa_chain