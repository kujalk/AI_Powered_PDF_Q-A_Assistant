import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

# Set page title
st.title("📄 PDF Q&A with OpenAI")

# Sidebar for API key input
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    st.markdown("[Get your OpenAI API key](https://platform.openai.com/account/api-keys)")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Initialize session state for processed data
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.text_chunks = None
    st.session_state.vectorstore = None

# Process the PDF when uploaded
if uploaded_file is not None and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if not st.session_state.processed:
        with st.spinner("Processing PDF..."):
            # 1. Read PDF text
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # 2. Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            text_chunks = text_splitter.split_text(text)
            st.session_state.text_chunks = text_chunks
            
            # 3. Create embeddings (vector representations of text)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(text_chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            
            st.session_state.processed = True
        st.success("PDF processed! You can now ask questions.")

# Question input
question = st.text_input("Ask a question about the PDF:")

# Answer the question
if question and st.session_state.processed:
    with st.spinner("Thinking..."):
        # 4. Search for relevant chunks
        docs = st.session_state.vectorstore.similarity_search(question)
        #print(f"Found {len(docs)} relevant chunks.")
        # Display the relevant chunks
        #print(f"Relevant chunks: {docs}")
        
        # 5. Use OpenAI to generate an answer
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=question)
            st.write(f"**Answer:** {response}")
            st.write(f"Cost: ${cb.total_cost:.4f}")