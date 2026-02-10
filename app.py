# app.py
import os
import streamlit as st
from PyPDF2 import PdfReader

# LangChain-ish helpers you already used
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A")

# --- Session state defaults
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- Upload
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in pdf_reader.pages]
        st.session_state.text = "\n\n".join(pages)
    else:
        st.session_state.text = uploaded_file.read().decode("utf-8")

    st.success(f"Document loaded: {len(st.session_state.text)} characters")

# --- Processing controls
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
top_k = st.sidebar.number_input("Top K documents for retrieval", min_value=1, max_value=10, value=4, step=1)

if st.button("Process Document") and st.session_state.text:
    with st.spinner("Splitting text and creating embeddings..."):
        # 1) split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(st.session_state.text)
        st.session_state.chunks = chunks

        # 2) create embeddings and vectorstore
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.success(f"Document processed into {len(chunks)} chunks and indexed.")
        except Exception as e:
            st.error(f"Failed to create embeddings or vectorstore: {e}")

# --- Require vectorstore for downstream tabs
if not st.session_state.vectorstore:
    st.info("Upload and process a document to enable Summary and Q&A.")
    st.stop()

# --- Hugging Face token
api_token = os.getenv("HUGGINGFACE_API_TOKEN") or st.secrets.get("HUGGINGFACE_API_TOKEN", None)
if not api_token:
    st.error("⚠️ HUGGINGFACE_API_TOKEN not found. Set it in environment or Streamlit secrets.")
    st.stop()

# --- Tabs for Summary and Q&A
tab1, tab2 = st.tabs(["📝 Summary", "❓ Q&A"])

with tab1:
    st.markdown("**Generate a short summary of the document.**")
    summary_max_input = st.slider("Max input characters to summarize (trim long docs)", 256, 8192, 2048, step=256)
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                # Use a summarization model endpoint
                llm = HuggingFaceEndpoint(
                    repo_id="facebook/bart-large-cnn",
                    huggingfacehub_api_token=api_token,
                    task="summarization"
                )
                # Trim input to model-friendly size
                input_text = st.session_state.text[:summary_max_input]
                raw = llm.invoke(input_text)  # wrapper may accept string or dict
                # Normalize response
                if isinstance(raw, dict):
                    summary = raw.get("summary_text") or raw.get("generated_text") or str(raw)
                else:
                    summary = str(raw)
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")

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
                context = "\n\n".join([d.page_content for d in docs])

                # 2) build a prompt
                prompt_template = (
                    "You are an assistant that answers questions using the provided context.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer concisely and cite context when helpful."
                )
                prompt_text = prompt_template.format(context=context, question=question)

                # 3) call the LLM endpoint
                llm = HuggingFaceEndpoint(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token=api_token
                )
                raw = llm.invoke(prompt_text)  # wrapper may accept string or dict

                # 4) normalize output
                if isinstance(raw, dict):
                    # common keys: 'generated_text', 'summary_text', 'output'
                    answer = raw.get("generated_text") or raw.get("summary_text") or raw.get("output") or str(raw)
                else:
                    answer = str(raw)

                st.subheader("Answer")
                st.write(answer)

                # optional: show retrieved snippets for transparency
                with st.expander("Retrieved snippets"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(f"**Snippet {i}**")
                        st.write(d.page_content[:1000])
            except Exception as e:
                st.error(f"Q&A failed: {e}")

# --- Footer
st.markdown("---")
st.caption("Notes  •  This app uses FAISS for vector search and Hugging Face endpoints for LLMs and embeddings.")
