import streamlit as st
from qa_chain import build_qa_chain

st.set_page_config(page_title="Retail Chatbot", layout="centered")
st.title("Retail Chatbot")

lang = st.radio("Choose language / Elige idioma", ["English", "Español"])

prompt = st.text_input("Ask a question / Haz una pregunta")

if prompt:
    qa_chain = build_qa_chain()
    response = qa_chain.run(prompt)
    st.markdown("### Response / Respuesta")
    st.write(response)
