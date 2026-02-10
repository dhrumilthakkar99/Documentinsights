import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains.retrieval_qa.base import RetrievalQA
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
    tab1, tab2 = st.tabs(["📝 Summary", "❓ Q&A"])
    
    with tab1:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn", 
                                     huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"))
                summary = llm(st.session_state.text[:1024])
                st.write(summary)
    
    with tab2:
        question = st.text_input("Ask a question about the document:")
        if question:
            with st.spinner("Finding answer..."):
                llm = HuggingFaceHub(repo_id="google/flan-t5-large",
                                     huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"))
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectorstore.as_retriever())
                answer = qa_chain.run(question)
                st.write(answer)
