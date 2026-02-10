# app.py
"""
Streamlit Document Summarization & Q&A (RAG)

Primary LLM route:
  Hugging Face Router -> Groq provider (OpenAI-compatible chat completions)

Fallback:
  Groq Direct API (OpenAI-compatible chat completions)

Requirements:
  pip install streamlit PyPDF2 langchain-text-splitters langchain-huggingface langchain-community faiss-cpu openai

Secrets / env:
  HUGGINGFACE_API_TOKEN  (required for HF Router)
  GROQ_API_KEY           (recommended for Groq fallback)

Notes:
- Uses FAISS for retrieval.
- Uses HuggingFaceEmbeddings locally for embeddings.
- Uses chat-completions to avoid provider "text-generation vs conversational" mismatches.
"""

import os
import traceback
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Document Q&A (HF→Groq + Groq fallback)", layout="wide")
st.title("📄 Document Summarization & Q&A (HF→Groq + Groq fallback)")


# ----------------------------
# Session state defaults
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False


# ----------------------------
# Helpers
# ----------------------------
def get_secret(name: str, default=None):
    """Fetch from env or Streamlit secrets."""
    return os.getenv(name) or st.secrets.get(name, default)


def hf_routed_model(model_id: str) -> str:
    """
    Force HF Router to use Groq provider.
    If model already has a provider suffix (e.g. ':groq'), keep it.
    """
    if ":" in model_id:
        return model_id
    return f"{model_id}:groq"


def normalize_text(s: str) -> str:
    return (s or "").strip()


def openai_chat(client: OpenAI, model: str, messages, temperature: float, max_tokens: int):
    """
    Minimal OpenAI-compatible chat call.
    Returns (content, full_response_object).
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content, resp


def llm_chat_with_fallback(
    model_id: str,
    messages,
    temperature: float,
    max_tokens: int,
    hf_token: str | None,
    groq_key: str | None,
    debug: bool = False,
):
    """
    Primary: HF Router -> Groq provider
      base_url: https://router.huggingface.co/v1
      model: <model_id>:groq

    Fallback: Groq direct
      base_url: https://api.groq.com/openai/v1
      model: <model_id>

    Returns dict with:
      { "content": str, "primary_used": bool, "raw": resp, "error_primary": str|None, "error_fallback": str|None }
    """
    result = {
        "content": "",
        "primary_used": False,
        "raw": None,
        "error_primary": None,
        "error_fallback": None,
    }

    # --- Primary (HF Router)
    if not hf_token:
        result["error_primary"] = "Missing HUGGINGFACE_API_TOKEN"
    else:
        try:
            hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            routed = hf_routed_model(model_id)
            content, raw = openai_chat(
                hf_client, routed, messages, temperature=temperature, max_tokens=max_tokens
            )
            if normalize_text(content):
                result["content"] = content
                result["primary_used"] = True
                result["raw"] = raw
                return result
            else:
                result["error_primary"] = "Primary returned empty content."
        except Exception as e:
            result["error_primary"] = f"{type(e).__name__}: {e}"
            if debug:
                st.write("Primary exception traceback:")
                st.code(traceback.format_exc())

    # --- Fallback (Groq direct)
    if not groq_key:
        result["error_fallback"] = "Missing GROQ_API_KEY (fallback not available)"
        return result

    try:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        content, raw = openai_chat(
            groq_client, model_id, messages, temperature=temperature, max_tokens=max_tokens
        )
        result["content"] = content
        result["raw"] = raw
        return result
    except Exception as e:
        result["error_fallback"] = f"{type(e).__name__}: {e}"
        if debug:
            st.write("Fallback exception traceback:")
            st.code(traceback.format_exc())
        return result


def build_qa_prompt(context_blocks: list[str], question: str) -> list[dict]:
    """
    Build a messages list for chat completion.
    We tag snippets as [S1], [S2], ... and ask model to cite them.
    """
    ctx = "\n\n".join(context_blocks)
    return [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Answer ONLY using the provided snippets.\n"
                "If the answer is not present, say: 'Not found in the document.'\n"
                "When you use a snippet, cite it like [S1], [S2]."
            ),
        },
        {
            "role": "user",
            "content": f"Snippets:\n{ctx}\n\nQuestion: {question}\nAnswer:",
        },
    ]


def build_summary_prompt(text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Summarize faithfully and avoid adding facts.\n"
                "Return 5-8 bullet points, concise and information-dense."
            ),
        },
        {
            "role": "user",
            "content": f"Summarize the following document:\n\n{text}",
        },
    ]


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
top_k = st.sidebar.number_input("Top K documents for retrieval", min_value=1, max_value=10, value=4, step=1)

st.sidebar.markdown("---")
st.sidebar.header("LLM Options (Primary: HF Router→Groq, Fallback: Groq Direct)")
model_id = st.sidebar.text_input("Model ID", value="openai/gpt-oss-20b")
max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, step=64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, step=0.1)

st.sidebar.markdown("---")
st.session_state.debug_raw = st.sidebar.checkbox("Show raw LLM response / debug", value=False)


# ----------------------------
# Upload area
# ----------------------------
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in pdf_reader.pages]
            st.session_state.text = "\n\n".join(pages)
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8", errors="replace")

        st.success(f"Document loaded: {len(st.session_state.text)} characters")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")


# ----------------------------
# Process document button
# ----------------------------
if st.button("Process Document") and st.session_state.text:
    with st.spinner("Splitting text and creating embeddings..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )
            chunks = text_splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            # Local embeddings (no HF inference)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)

            st.success(f"Document processed into {len(chunks)} chunks and indexed.")
        except Exception as e:
            st.error(f"Failed to create embeddings or vectorstore: {e}")


# ----------------------------
# Require vectorstore
# ----------------------------
if not st.session_state.vectorstore:
    st.info("Upload and process a document to enable Summary and Q&A.")
    st.stop()


# ----------------------------
# Tokens
# ----------------------------
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)

if not hf_token:
    st.warning("⚠️ HUGGINGFACE_API_TOKEN not found. Primary route (HF Router→Groq) will not work.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY not found. Fallback (Groq Direct) will not work if primary fails.")


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📝 Summary", "❓ Q&A"])


# ----------------------------
# Summary tab
# ----------------------------
with tab1:
    st.markdown("**Generate a short summary of the document.**")

    # For long docs, optionally do a lightweight map-reduce style summary using the already-created chunks.
    use_chunked_summary = st.checkbox("Use chunked summary (better for long documents)", value=True)
    max_input_chars = st.slider("Max characters (non-chunked summary)", 256, 50000, 5000, step=256)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                if use_chunked_summary and st.session_state.chunks:
                    # Summarize first N chunks, then combine.
                    # Keep limits conservative to reduce cost/latency.
                    n_chunks = min(8, len(st.session_state.chunks))
                    mini_summaries = []

                    for i in range(n_chunks):
                        chunk_text = st.session_state.chunks[i]
                        messages = [
                            {"role": "system", "content": "Summarize the text in 2-3 bullet points. Avoid adding facts."},
                            {"role": "user", "content": chunk_text},
                        ]
                        out = llm_chat_with_fallback(
                            model_id=model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=min(256, max_tokens),
                            hf_token=hf_token,
                            groq_key=groq_key,
                            debug=st.session_state.debug_raw,
                        )
                        mini_summaries.append(out["content"] or "")

                    final_messages = [
                        {
                            "role": "system",
                            "content": "Combine the bullet points into a single 6-10 bullet executive summary. No extra facts.",
                        },
                        {"role": "user", "content": "\n\n".join(mini_summaries)},
                    ]

                    result = llm_chat_with_fallback(
                        model_id=model_id,
                        messages=final_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        hf_token=hf_token,
                        groq_key=groq_key,
                        debug=st.session_state.debug_raw,
                    )
                else:
                    input_text = st.session_state.text[: int(max_input_chars)]
                    messages = build_summary_prompt(input_text)

                    result = llm_chat_with_fallback(
                        model_id=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        hf_token=hf_token,
                        groq_key=groq_key,
                        debug=st.session_state.debug_raw,
                    )

                if st.session_state.debug_raw:
                    st.subheader("Routing / Debug")
                    st.write(
                        {
                            "primary_used": result["primary_used"],
                            "model_primary": hf_routed_model(model_id),
                            "model_fallback": model_id,
                            "error_primary": result["error_primary"],
                            "error_fallback": result["error_fallback"],
                        }
                    )
                    st.subheader("Raw LLM Response")
                    st.write(result["raw"])

                summary = normalize_text(result["content"])
                if summary:
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("No summary returned.")
                    st.write(
                        {
                            "error_primary": result["error_primary"],
                            "error_fallback": result["error_fallback"],
                        }
                    )

            except Exception as e:
                st.error("Summarization failed.")
                st.exception(e)


# ----------------------------
# Q&A tab
# ----------------------------
with tab2:
    st.markdown("**Ask a question about the document.**")
    question = st.text_input("Question", "")

    if st.button("Get Answer") and question.strip():
        with st.spinner("Finding answer..."):
            try:
                # Retrieve top-k chunks
                docs = st.session_state.vectorstore.similarity_search(question, k=int(top_k))
                if not docs:
                    st.warning("No relevant documents found.")
                    st.stop()

                # Tag snippets
                context_blocks = []
                for i, d in enumerate(docs, start=1):
                    snippet = d.page_content.strip()
                    context_blocks.append(f"[S{i}] {snippet}")

                messages = build_qa_prompt(context_blocks, question.strip())

                result = llm_chat_with_fallback(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    hf_token=hf_token,
                    groq_key=groq_key,
                    debug=st.session_state.debug_raw,
                )

                if st.session_state.debug_raw:
                    st.subheader("Routing / Debug")
                    st.write(
                        {
                            "primary_used": result["primary_used"],
                            "model_primary": hf_routed_model(model_id),
                            "model_fallback": model_id,
                            "error_primary": result["error_primary"],
                            "error_fallback": result["error_fallback"],
                        }
                    )
                    st.subheader("Raw LLM Response")
                    st.write(result["raw"])

                answer = normalize_text(result["content"])
                st.subheader("Answer")
                st.write(answer if answer else "No answer returned.")

                with st.expander("Retrieved snippets"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(f"**Snippet {i}**")
                        st.write(d.page_content[:1500])

            except Exception as e:
                st.error("Q&A failed.")
                st.exception(e)


# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Notes • FAISS for vector search • HuggingFaceEmbeddings locally • Chat-completions via HF Router→Groq with Groq Direct fallback.")
