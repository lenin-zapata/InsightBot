# agent.py
import os
import re
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Importamos lógica de tus otros archivos
from vector_store import load_dataframe
from qa_chain import build_qa_chain

# ==========================================
# 1. AUTENTICACIÓN (Fix para Streamlit Cloud)
# ==========================================
# Si el token está en los secretos de Streamlit, lo pasamos al entorno
# para que LangChain y HuggingFace lo detecten automáticamente.
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# =========================
# Helpers de datos y parsing
# =========================
@st.cache_resource(show_spinner=False)
def get_df() -> pd.DataFrame:
    return load_dataframe()

def _detect_metric(text: str) -> str:
    t = text.lower()
    if "projected" in t or "proyectad" in t:
        return "Projected Sales"
    if "purchase" in t or "compra" in t:
        return "Actual Purchases"
    return "Actual Sales"

def _detect_groupby(text: str) -> str:
    t = text.lower()
    if "month" in t or "mes" in t or "mensual" in t:
        return "Month/Year"
    if "category" in t or "categoría" in t or "categoria" in t:
        return "Category"
    if "product" in t or "producto" in t:
        return "Product"
    return "Month/Year"

def _extract_product(text: str) -> str | None:
    m = re.search(r"(?:product|producto)\s*[:=]?\s*['\"]?([A-Za-z0-9 \-\_/]+?)['\"]?(?:$|,|;)", text, re.I)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"'\"['\"]", text)
    return m2.group(1).strip() if m2 else None

def _extract_topk(text: str) -> int | None:
    m = re.search(r"\btop[-\s]?(\d+)\b", text, re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b(?:los|the)\s+(\d+)\s+mejores\b", text, re.I)
    return int(m2.group(1)) if m2 else None


# =========================
# Tools: cálculos y gráficos
# =========================
def tool_retail_calc(nl_query: str) -> str:
    """
    Cálculos sobre el DataFrame (suma, promedio, top-k).
    Filtro opcional por 'producto: <nombre>'.
    """
    df = get_df().copy()
    metric = _detect_metric(nl_query)
    product = _extract_product(nl_query)
    topk = _extract_topk(nl_query)

    if product:
        df = df[df["Product"].str.lower() == product.lower()]

    if df.empty:
        return "No hay datos que coincidan con el filtro aplicado."

    t = nl_query.lower()
    if any(k in t for k in ["promedio", "average", "mean"]):
        val = float(df[metric].mean())
        return f"Promedio de **{metric}**{' para '+product if product else ''}: {val:,.2f}"
    if any(k in t for k in ["total", "suma", "sum"]):
        val = float(df[metric].sum())
        return f"Total de **{metric}**{' para '+product if product else ''}: {val:,.2f}"

    if (("top" in t or "mejores" in t or "más vendidos" in t or "mas vendidos" in t)
        and ("product" in t or "producto" in t or "productos" in t)):
        k = topk or 5
        agg = (
            df.groupby("Product", as_index=False)[metric]
              .sum()
              .sort_values(metric, ascending=False)
              .head(k)
        )
        lines = [f"Top {k} productos por **{metric}**:"]
        for i, row in enumerate(agg.itertuples(index=False), start=1):
            lines.append(f"{i}. {row.Product}: {getattr(row, metric):,.2f}")
        return "\n".join(lines)

    desc = df[metric].describe().to_dict()
    return f"Estadísticos de **{metric}**{' para '+product if product else ''}: {desc}"

def tool_retail_plot(nl_query: str) -> str:
    """
    Genera un gráfico (líneas/barras) según intención.
    Devuelve 'PLOT_PATH: <ruta>' + una línea 'Gráfico generado: ...'.
    """
    df = get_df().copy()
    metric = _detect_metric(nl_query)
    groupby = _detect_groupby(nl_query)
    product = _extract_product(nl_query)

    if product:
        df = df[df["Product"].str.lower() == product.lower()]

    if df.empty:
        return "No hay datos que coincidan con el filtro aplicado."

    agg = df.groupby(groupby, as_index=False)[metric].sum()

    if groupby == "Month/Year":
        try:
            agg["_dt"] = pd.to_datetime(agg[groupby], infer_datetime_format=True, errors="coerce")
            if agg["_dt"].notna().any():
                agg = agg.sort_values("_dt")
        except Exception:
            pass

    plt.figure(figsize=(8, 4.5))
    if groupby == "Month/Year":
        plt.plot(agg[groupby], agg[metric], marker="o")
        plt.xticks(rotation=45, ha="right")
    else:
        agg = agg.sort_values(metric, ascending=False)
        plt.bar(agg[groupby].astype(str), agg[metric])
        plt.xticks(rotation=45, ha="right")

    title = f"{metric} por {groupby}" + (f" – {product}" if product else "")
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()

    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, "retail_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return f"PLOT_PATH: {out_path}\nGráfico generado: {title}"


# =========================
# Tool: RAG (envoltura)
# =========================
def build_rag_tool(qa_chain, target_lang: str) -> Tool:
    """
    Envuelve la cadena RAG como herramienta para el agente.
    Devuelve respuesta + fuentes (si existen).
    """
    def _rag_fn(question: str) -> str:
        result = qa_chain.invoke({
            "question": question,
            "chat_history": [],      # la memoria la maneja el agente
            "target_lang": target_lang
        })
        answer = (result.get("answer") or "").strip()
        src = result.get("source_documents") or []
        if src:
            lines = []
            for i, d in enumerate(src, start=1):
                meta = d.metadata or {}
                tag = f"{meta.get('category', '')} / {meta.get('product', '')} / {meta.get('month_year', '')}"
                lines.append(f"- Fuente {i}: {tag}".strip())
            answer += "\n\n**Fuentes:**\n" + "\n".join(lines)
        return answer

    return Tool(
        name="Dataset QA",
        description=("Usa esta herramienta para responder preguntas sobre el dataset retail "
                     "(ventas, compras, proyecciones) usando RAG."),
        func=_rag_fn
    )


# =========================
# Construcción del agente
# =========================
@st.cache_resource(show_spinner=False)
def build_agent(model_name: str = "HuggingFaceH4/zephyr-7b-beta",
                target_lang: str = "Español"):
    """
    Agente conversacional (CHAT_CONVERSATIONAL_REACT_DESCRIPTION) con:
    - Dataset QA (RAG)
    - RetailCalc (cálculos)
    - RetailPlot (gráficos)
    - DuckDuckGoSearchRun (búsqueda web)
    Servido por Hugging Face Inference API.
    """
    
    # Obtenemos el token de las variables de entorno (ya inyectado desde st.secrets arriba)
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # LLM chat via HF Inference API
    hf_raw = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
        timeout=120,
        huggingfacehub_api_token=api_token, # Token explícito
    )
    llm = ChatHuggingFace(llm=hf_raw)

    # RAG tool
    qa_chain = build_qa_chain(model_name)
    rag_tool = build_rag_tool(qa_chain, target_lang)

    # Otras tools
    calc_tool = Tool(
        name="RetailCalc",
        description=("Cálculos sobre el dataset (totales, promedios, top-k). "
                     "Puedes filtrar con 'producto: <nombre>'."),
        func=tool_retail_calc
    )
    plot_tool = Tool(
        name="RetailPlot",
        description=("Genera gráficos sencillos (líneas/barras). Devuelve PLOT_PATH para mostrar en la UI."),
        func=tool_retail_plot
    )
    web_tool = DuckDuckGoSearchRun()

    tools = [rag_tool, calc_tool, plot_tool, web_tool]

    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True
    )

    prefix = f"""
Eres un asistente de analítica retail. Responde SIEMPRE en {target_lang}.
Política de herramientas:
1) Prefiere "Dataset QA" cuando la información esté en el dataset.
2) Usa "RetailCalc" para cálculos numéricos.
3) Usa "RetailPlot" si piden gráficos.
4) Usa "DuckDuckGoSearchRun" sólo si el dataset no tiene la información.
5) Si faltan datos, dilo y sugiere qué falta.
""".strip()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        agent_kwargs={"prefix": prefix},
        handle_parsing_errors=True,  # tolera desvíos menores de formato
    )
    return agent