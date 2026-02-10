# app.py
"""
Streamlit Document Summarization & Q&A

Requirements:
pip install streamlit PyPDF2 langchain-text-splitters langchain-huggingface langchain-community faiss-cpu

Set HUGGINGFACE_API_TOKEN in your environment or Streamlit secrets.
"""

import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

# --- Page config
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A")

# --- Session state defaults
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False

# --- Sidebar controls
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
top_k = st.sidebar.number_input("Top K documents for retrieval", min_value=1, max_value=10, value=4, step=1)
st.sidebar.markdown("---")
st.session_state.debug_raw = st.sidebar.checkbox("Show raw LLM response", value=False)

# --- Upload area
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in pdf_reader.pages]
            st.session_state.text = "\n\n".join(pages)
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8")
        st.success(f"Document loaded: {len(st.session_state.text)} characters")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

# --- Process document button
if st.button("Process Document") and st.session_state.text:
    with st.spinner("Splitting text and creating embeddings..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.success(f"Document processed into {len(chunks)} chunks and indexed.")
        except Exception as e:
            st.error(f"Failed to create embeddings or vectorstore: {e}")

# --- Require vectorstore for downstream
if not st.session_state.vectorstore:
    st.info("Upload and process a document to enable Summary and Q&A.")
    st.stop()

# --- Hugging Face token
api_token = os.getenv("HUGGINGFACE_API_TOKEN") or st.secrets.get("HUGGINGFACE_API_TOKEN", None)
if not api_token:
    st.error("⚠️ HUGGINGFACE_API_TOKEN not found. Set it in environment or Streamlit secrets.")
    st.stop()

# --- Tabs
tab1, tab2 = st.tabs(["📝 Summary", "❓ Q&A"])

# --- Helper to normalize LLM responses
def normalize_llm_response(raw):
    """
    Normalize common Hugging Face endpoint return shapes to a string.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # common keys
        for key in ("summary_text", "generated_text", "output", "text", "result"):
            if key in raw and raw[key]:
                return raw[key]
        # sometimes the response is nested or a list
        # try to stringify useful parts
        if "error" in raw:
            return f"ERROR: {raw['error']}"
        return str(raw)
    if isinstance(raw, (list, tuple)):
        # join simple lists
        try:
            return "\n".join([normalize_llm_response(r) for r in raw])
        except Exception:
            return str(raw)
    return str(raw)

# --- Summary tab
with tab1:
    st.markdown("**Generate a short summary of the document.**")
    summary_max_input = st.slider("Max input characters to summarize (trim long docs)", 256, 8192, 2048, step=256)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            input_text = st.session_state.text[:summary_max_input]

            # Build a string prompt (HuggingFaceEndpoint.invoke expects a string) [1](https://docs.langchain.com/oss/python/integrations/llms/huggingface_endpoint)[2](https://sj-langchain.readthedocs.io/en/latest/llms/langchain.llms.huggingface_endpoint.HuggingFaceEndpoint.html)
            prompt = (
                "Summarize the following text in 3–5 bullet points. "
                "Focus on key ideas and outcomes.\n\n"
                f"{input_text}"
            )

            raw = None
            try:
                # Try an instruction model (more robust than forcing summarization task)
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    huggingfacehub_api_token=api_token,
                    temperature=0.2,
                    max_new_tokens=256,
                    # provider="auto",  # optional: set a specific provider if needed
                )
                raw = llm.invoke(prompt)  # string only
            except Exception as e:
                st.error("Summarization failed (Mistral). Showing exception:")
                st.exception(e)

            if st.session_state.debug_raw:
                st.subheader("Raw LLM Response")
                st.write(raw)

            summary = normalize_llm_response(raw)
            if summary:
                st.subheader("Summary")
                st.write(summary)
            else:
                st.warning("No summary returned. Enable debug and check the exception output above.")
                
# --- Q&A tab
with tab2:
    st.markdown("**Ask a question about the document.**")
    question = st.text_input("Question", "")
    if st.button("Get Answer") and question.strip():
        with st.spinner("Finding answer..."):
            try:
                # 1) retrieve top-k relevant chunks
                docs = st.session_state.vectorstore.similarity_search(question, k=int(top_k))
                if not docs:
                    st.warning("No relevant documents found.")
                    st.stop()
                context = "\n\n".join([d.page_content for d in docs])

                # 2) build prompt
                prompt_template = (
                    "You are an assistant that answers questions using the provided context.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer concisely and cite context when helpful."
                )
                prompt_text = prompt_template.format(context=context, question=question)

                # 3) call the LLM endpoint (instruction-following model)
                llm = HuggingFaceEndpoint(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token=api_token
                )

                # Try string first, then dict if wrapper expects inputs key
                try:
                    raw = llm.invoke(prompt_text)
                except Exception:
                    raw = llm.invoke({prompt_text})

                if st.session_state.debug_raw:
                    st.subheader("Raw LLM Response")
                    st.write(raw)

                answer = normalize_llm_response(raw)
                st.subheader("Answer")
                st.write(answer)

                # show retrieved snippets for transparency
                with st.expander("Retrieved snippets"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(f"**Snippet {i}**")
                        st.write(d.page_content[:1000])
            except Exception as e:
                st.error(f"Q&A failed: {e}")

# --- Footer
st.markdown("---")
st.caption("Notes  •  This app uses FAISS for vector search and Hugging Face endpoints for LLMs and embeddings.")
