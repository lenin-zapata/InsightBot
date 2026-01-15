import os
import streamlit as st
from dotenv import load_dotenv
import html 

# =========================
# Configuración inicial y Secretos
# =========================
st.set_page_config(page_title="Retail Agent", layout="centered")

# 1. Cargar variables de entorno locales (.env)
load_dotenv()

# 2. Inyectar secretos de Streamlit al entorno (CRÍTICO para la Nube)
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Ahora importamos el agente (después de configurar el entorno)
from agent import build_agent

st.title("🛍️ Retail Agent (RAG + Tools + Conversación)")

# =========================
# Barra lateral (settings)
# =========================
st.sidebar.header("⚙️ Configuración")

# Modelos instruct/chat servidos por Hugging Face Inference API
model_choice = st.sidebar.selectbox(
    "Modelo (HF Inference - instruct/chat)",
    [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
    ],
    index=0,
)

lang = st.sidebar.radio("Idioma de respuesta / Response language", ["Español", "English"], index=0)

# Validación de token
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.sidebar.error(
        "⚠️ No se encontró el Token de Hugging Face. "
        "Configúralo en los 'Secrets' de Streamlit Cloud o en tu .env local."
    )
    st.stop() # Detiene la app si no hay token

st.sidebar.caption(
    "Consejo: si cambias el CSV, borra la carpeta `.faiss_index/` para regenerar el índice."
)

# =========================
# Estado de sesión
# =========================
def init_agent():
    st.session_state.agent = build_agent(model_choice, target_lang=lang)
    st.session_state.model_name = model_choice
    st.session_state.target_lang = lang

if "agent" not in st.session_state:
    init_agent()
elif st.session_state.get("model_name") != model_choice or st.session_state.get("target_lang") != lang:
    init_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# Render del historial
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("plot_path"):
            st.markdown(m["content"])
            if os.path.exists(m["plot_path"]):
                st.image(m["plot_path"], caption=m.get("caption", ""), use_container_width=True)
        else:
            st.markdown(m["content"])

# =========================
# Controles superiores
# =========================
cols = st.columns(2)
with cols[0]:
    if st.button("🧹 Reiniciar conversación"):
        init_agent()
        st.session_state.messages = []
        st.rerun()
with cols[1]:
    st.info("Pide análisis o gráficos; el agente elegirá la herramienta adecuada.")

# =========================
# Utilidad para detectar PLOT_PATH
# =========================
def maybe_render_plot(text: str):
    lines = text.splitlines()
    path = None
    caption = None
    for ln in lines:
        if "PLOT_PATH:" in ln:
            path = ln.split("PLOT_PATH:", 1)[1].strip()
        if ln.lower().startswith("gráfico generado") or ln.lower().startswith("grafico generado"):
            caption = ln.strip()
    return path, caption

# =========================
# Entrada del usuario
# =========================
user_input = st.chat_input("Pregunta o pide análisis/gráficos...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                result = st.session_state.agent.invoke({"input": user_input})
                answer = result["output"].strip() if isinstance(result, dict) else str(result).strip()
                answer = html.unescape(answer)

                plot_path, caption = maybe_render_plot(answer)
                
                # Render logic
                if plot_path and os.path.exists(plot_path):
                    answer_clean = "\n".join([ln for ln in answer.splitlines() if "PLOT_PATH:" not in ln])
                    st.markdown(answer_clean)
                    st.image(plot_path, caption=caption or "Gráfico", use_container_width=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_clean,
                        "plot_path": plot_path,
                        "caption": caption or ""
                    })
                else:
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error al procesar: {e}")