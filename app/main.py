import streamlit as st
import os
from dotenv import load_dotenv
from qa_chain import build_qa_chain

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.set_page_config(page_title="Retail Chatbot", layout="centered")
st.title("Retail Chatbot")


model_choice = st.selectbox(
    "Choose model size / Elige tamaño del modelo",
    ["flan-t5-small", "flan-t5-base", "flan-t5-large"]
)

if model_choice == "flan-t5-large":
    st.warning("Este modelo puede tardar más en responder pero tiene mayor precisión.")

lang = st.radio("Choose language / Elige idioma", ["English", "Español"])

prompt = st.text_input(
    "Ask a question / Haz una pregunta",
    placeholder="e.g. What are the top-selling products this month? / ¿Cuáles son los productos más vendidos este mes?"
)


qa_chain = build_qa_chain(model_name=model_choice)

if prompt:
    try:
        response = qa_chain.invoke(prompt)
        st.markdown("### Response / Respuesta")
        st.write(response)
    except Exception as e:
        st.error(f"Error al procesar la pregunta: {e}")
