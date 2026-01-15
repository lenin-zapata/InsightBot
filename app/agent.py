# agent.py
import os
import re
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# CAMBIO: Usamos langchain_community en lugar de langchain_huggingface
from langchain_community.llms import HuggingFaceEndpoint
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Importamos lógica de tus otros archivos
from vector_store import load_dataframe
from qa_chain import build_qa_chain

# ==========================================
# 1. AUTENTICACIÓN
# ==========================================
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# =========================
# Helpers (se mantienen igual)
# =========================
@st.cache_resource(show_spinner=False)
def get_df() -> pd.DataFrame:
    return load_dataframe()

def _detect_metric(text: str) -> str:
    t = text.lower()
    if "projected" in t or "proyectad" in t: return "Projected Sales"
    if "purchase" in t or "compra" in t: return "Actual Purchases"
    return "Actual Sales"

def _detect_groupby(text: str) -> str:
    t = text.lower()
    if "month" in t or "mes" in t: return "Month/Year"
    if "category" in t or "categoría" in t: return "Category"
    if "product" in t or "producto" in t: return "Product"
    return "Month/Year"

def _extract_product(text: str) -> str | None:
    m = re.search(r"(?:product|producto)\s*[:=]?\s*['\"]?([A-Za-z0-9 \-\_/]+?)['\"]?(?:$|,|;)", text, re.I)
    return m.group(1).strip() if m else None

def _extract_topk(text: str) -> int | None:
    m = re.search(r"\btop[-\s]?(\d+)\b", text, re.I)
    return int(m.group(1)) if m else None

# =========================
# Tools (se mantienen igual)
# =========================
def tool_retail_calc(nl_query: str) -> str:
    df = get_df().copy()
    metric = _detect_metric(nl_query)
    product = _extract_product(nl_query)
    topk = _extract_topk(nl_query)

    if product:
        df = df[df["Product"].str.lower() == product.lower()]
    if df.empty: return "No hay datos para ese filtro."

    t = nl_query.lower()
    if any(k in t for k in ["promedio", "average"]):
        return f"Promedio de {metric}: {df[metric].mean():,.2f}"
    if any(k in t for k in ["total", "suma"]):
        return f"Total de {metric}: {df[metric].sum():,.2f}"
    
    if "top" in t:
        k = topk or 5
        agg = df.groupby("Product", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(k)
        lines = [f"Top {k} productos por {metric}:"]
        for i, row in enumerate(agg.itertuples(index=False), start=1):
            lines.append(f"{i}. {row.Product}: {getattr(row, metric):,.2f}")
        return "\n".join(lines)

    return f"Estadísticas: {df[metric].describe().to_dict()}"

def tool_retail_plot(nl_query: str) -> str:
    df = get_df().copy()
    metric = _detect_metric(nl_query)
    groupby = _detect_groupby(nl_query)
    product = _extract_product(nl_query)

    if product:
        df = df[df["Product"].str.lower() == product.lower()]
    
    agg = df.groupby(groupby, as_index=False)[metric].sum()
    
    plt.figure(figsize=(8, 4.5))
    if groupby == "Month/Year":
        plt.plot(agg[groupby], agg[metric], marker="o")
    else:
        agg = agg.sort_values(metric, ascending=False)
        plt.bar(agg[groupby].astype(str), agg[metric])
    
    plt.title(f"{metric} por {groupby}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, "retail_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return f"PLOT_PATH: {out_path}\nGráfico generado."

def build_rag_tool(qa_chain, target_lang: str) -> Tool:
    def _rag_fn(question: str) -> str:
        result = qa_chain.invoke({"question": question, "chat_history": [], "target_lang": target_lang})
        return result.get("answer", "").strip()
    return Tool(name="Dataset QA", description="Responde preguntas sobre retail usando RAG.", func=_rag_fn)

# =========================
# Construcción del agente
# =========================
@st.cache_resource(show_spinner=False)
def build_agent(model_name: str = "HuggingFaceH4/zephyr-7b-beta", target_lang: str = "Español"):
    
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # CAMBIO: Usamos HuggingFaceEndpoint directo (sin ChatHuggingFace)
    # Esto evita la necesidad de transformers/torch local
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=api_token,
    )

    qa_chain = build_qa_chain(model_name)
    rag_tool = build_rag_tool(qa_chain, target_lang)
    
    tools = [
        rag_tool, 
        Tool(name="RetailCalc", func=tool_retail_calc, description="Cálculos numéricos"),
        Tool(name="RetailPlot", func=tool_retail_plot, description="Gráficos"),
        DuckDuckGoSearchRun()
    ]

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    # CAMBIO: Usamos CONVERSATIONAL_REACT_DESCRIPTION (funciona mejor con LLMs puros)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
    )
    return agent