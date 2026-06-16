# Databricks notebook source
import streamlit as st
import os, json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.llms import Databricks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

CATALOG  = "workspace"
SCHEMA   = "default"
VOLUME   = "genai_doc_assistant"
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
FAISS_PATH  = f"{VOL_ROOT}/faiss_index"
DOCS_PATH   = f"{VOL_ROOT}/docs_output"

st.set_page_config(page_title="Pipeline Doc Assistant", page_icon="🔍", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0D1B2A; }
.src-card {
    background: #142233; border-left: 3px solid #00D4FF;
    padding: 6px 12px; margin: 4px 0; border-radius: 3px; font-size: 12px;
}
.answer-box {
    background: #142233; border: 1px solid #1A2E4A;
    padding: 16px; border-radius: 6px; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chains():
    embeddings  = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
    vectorstore = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    try:
        llm = Databricks(
            endpoint_name="databricks-meta-llama-3-1-70b-instruct",
            max_tokens=1024, temperature=0
        )
        _ = llm.invoke("OK")
        llm_name = "LLaMA 3.1 70B"
    except:
        llm = Databricks(
            endpoint_name="databricks-meta-llama-3-1-8b-instruct",
            max_tokens=1024, temperature=0
        )
        llm_name = "LLaMA 3.1 8B"

    QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a senior data engineering assistant.
Use ONLY the pipeline metadata in context. Be specific — name pipelines and tables.
If context is insufficient, say so.

Context:
{context}

Question: {question}
Answer:"""
    )

    DOC_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a technical writer for a data engineering team.
Generate complete Markdown documentation with sections:
# [Pipeline Name]
## Overview
## Data Flow
## Tasks
## Run History
## Tags
Use specific names from metadata. No placeholders.
Metadata: {context}
Request: {question}
Documentation:"""
    )

    CODE_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a senior Databricks engineer doing a code review.
Check for: collect() anti-patterns, missing partitioning, null handling gaps,
wrong write modes, missing OPTIMIZE/ZORDER, hardcoded paths.
Pipeline context: {context}
Code: {question}
Review (Critical / Warnings / Suggestions / Corrected code):"""
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    doc_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": DOC_PROMPT},
        return_source_documents=False
    )
    code_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": CODE_PROMPT},
        return_source_documents=False
    )

    return qa_chain, doc_chain, code_chain, vectorstore.index.ntotal, llm_name


# ── Header ────────────────────────────────────────────────────
st.title("🔍 Pipeline Doc Assistant")
st.caption("RAG-powered Q&A, documentation generation, and code review for data pipelines")

with st.spinner("Loading vector store and LLM..."):
    try:
        qa_chain, doc_chain, code_chain, vec_count, llm_name = load_chains()
        st.success(f"✅ Ready — {vec_count} vectors indexed · {llm_name}")
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("", ["💬 Q&A", "📄 Generate Doc", "🔍 Code Review"])

    st.divider()
    st.subheader("Try these")
    examples = [
        "What tables feed into gold_customer_360?",
        "What does silver_transform_customers do?",
        "Full lineage from raw to gold?",
        "Which pipelines are in bronze layer?",
        "Did any pipeline fail recently?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    st.caption(f"LLM: {llm_name}")
    st.caption(f"Vectors: {vec_count}")
    st.caption("Store: FAISS on UC Volume")
    st.caption("Embed: BGE-large-en (1024d)")

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources — {len(msg['sources'])} chunks retrieved"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="src-card">'
                        f'<b>{s["pipeline"]}</b> · {s["type"]} · layer: {s["layer"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ── Input ─────────────────────────────────────────────────────
hints = {
    "💬 Q&A":          "Ask anything about your pipelines...",
    "📄 Generate Doc": "Enter pipeline name e.g. gold_customer_360",
    "🔍 Code Review":  "Paste your PySpark code here...",
}
prefill  = st.session_state.pop("prefill", "")
user_in  = st.chat_input(hints.get(mode, "Type here..."))
if prefill and not user_in:
    user_in = prefill

if user_in:
    st.session_state.messages.append({"role": "user", "content": user_in})
    with st.chat_message("user"):
        st.markdown(user_in)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources_meta = []

            if mode == "💬 Q&A":
                result  = qa_chain.invoke({"query": user_in})
                answer  = result["result"]
                for doc in result.get("source_documents", []):
                    sources_meta.append({
                        "pipeline": doc.metadata.get("pipeline_name", "?"),
                        "type":     doc.metadata.get("chunk_type", "?"),
                        "layer":    doc.metadata.get("layer", "?"),
                    })

            elif mode == "📄 Generate Doc":
                result = doc_chain.invoke({
                    "query": f"Generate complete documentation for pipeline: {user_in}"
                })
                answer = result["result"]

            else:  # Code Review
                result = code_chain.invoke({"query": user_in})
                answer = result["result"]

        st.markdown(answer)

        if sources_meta:
            with st.expander(f"Sources — {len(sources_meta)} chunks retrieved"):
                for s in sources_meta:
                    st.markdown(
                        f'<div class="src-card">'
                        f'<b>{s["pipeline"]}</b> · {s["type"]} · layer: {s["layer"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources_meta
    })

# ── Footer metrics ────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vectors",    vec_count)
c2.metric("LLM",        llm_name.split(" ")[-1])
c3.metric("Vector DB",  "FAISS")
c4.metric("Embed Dims", "1024")
