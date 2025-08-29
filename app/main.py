import os
import streamlit as st
from dotenv import load_dotenv
from agent import build_agent
import html 


# =========================
# Configuración e iniciales
# =========================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

st.set_page_config(page_title="Retail Agent", layout="centered")
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

# Avisos de entorno
if not HF_TOKEN:
    st.sidebar.error(
        "No se encontró `HUGGINGFACEHUB_API_TOKEN` en el entorno. "
        "Crea un archivo `.env` con tu token de Hugging Face."
    )

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
        # Mensaje con gráfico (si existiera)
        if m.get("plot_path"):
            st.markdown(m["content"])
            st.image(m["plot_path"], caption=m.get("caption", ""), use_container_width=True)
        else:
            st.markdown(m["content"])

# =========================
# Controles superiores
# =========================
cols = st.columns(2)
with cols[0]:
    if st.button("🧹 Reiniciar conversación"):
        init_agent()                # reconstruye agente (reinicia memoria)
        st.session_state.messages = []
        st.rerun()
with cols[1]:
    st.info("Pide análisis o gráficos; el agente elegirá la herramienta adecuada.")

# =========================
# Utilidad para detectar PLOT_PATH en respuestas
# =========================
def maybe_render_plot(text: str):
    """
    Si la respuesta del agente incluye 'PLOT_PATH: <ruta>', devolvemos (ruta, caption).
    La UI mostrará la imagen y ocultará la línea con PLOT_PATH.
    """
    lines = text.splitlines()
    path = None
    caption = None
    for ln in lines:
        if ln.startswith("PLOT_PATH:"):
            path = ln.split("PLOT_PATH:", 1)[1].strip()
        # Captions sugeridos por la herramienta (línea que empieza con 'Gráfico generado')
        if ln.lower().startswith("gráfico generado") or ln.lower().startswith("grafico generado"):
            caption = ln.strip()
    return path, caption

# =========================
# Entrada del usuario
# =========================
user_input = st.chat_input("Pregunta o pide análisis/gráficos...")

if user_input:
    # Muestra mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            
            try:
                result = st.session_state.agent.invoke({"input": user_input})
                answer = result["output"].strip() if isinstance(result, dict) else str(result).strip()
                answer = html.unescape(answer)  # 🔧 decodifica entidades HTML

                # ¿Incluye gráfico?
                plot_path, caption = maybe_render_plot(answer)
                if plot_path and os.path.exists(plot_path):
                    answer_clean = "\n".join([ln for ln in answer.splitlines() if not ln.startswith("PLOT_PATH:")])
                    answer_clean = html.unescape(answer_clean)  # 🔧 decodifica también el texto limpio
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

