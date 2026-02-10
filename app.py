import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
import os

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "text" not in st.session_state:
    st.session_state.text = ""

uploaded_file = st.file_uploader("Upload Document (PDF/TXT)", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        st.session_state.text = "".join([page.extract_text() for page in pdf_reader.pages])
    else:
        st.session_state.text = uploaded_file.read().decode("utf-8")
    
    st.success(f"Document loaded: {len(st.session_state.text)} characters")
    
    if st.button("Process Document"):
        with st.spinner("Processing..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(st.session_state.text)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.success("Document processed!")

if st.session_state.vectorstore:
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not api_token:
        st.error("⚠️ HUGGINGFACE_API_TOKEN not found. Please set it in Streamlit secrets.")
    else:
        tab1, tab2 = st.tabs(["📝 Summary", "❓ Q&A"])
        
        with tab1:
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    llm = HuggingFaceHub(
                        repo_id="facebook/bart-large-cnn",
                        huggingfacehub_api_token=api_token,
                        model_kwargs={"temperature": 0.5, "max_length": 130}
                    )
                    summary = llm.invoke(st.session_state.text[:1024])
                    st.write(summary)
        
        with tab2:
            question = st.text_input("Ask a question about the document:")
            if question:
                with st.spinner("Finding answer..."):
                    llm = HuggingFaceHub(
                        repo_id="google/flan-t5-large",
                        huggingfacehub_api_token=api_token
                    )
                    prompt = PromptTemplate.from_template("Answer based on context:\n{context}\n\nQuestion: {question}")
                    qa_chain = (
                        {"context": st.session_state.vectorstore.as_retriever(), "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    answer = qa_chain.invoke(question)
                    st.write(answer)
