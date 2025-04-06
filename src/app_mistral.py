import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Changed this
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama  # Changed this
import requests
import json

# Set page title
st.title("📄 PDF Q&A (OpenAI or Local Mistral)")

# Sidebar for model selection
with st.sidebar:
    st.header("Settings")
    use_local_mistral = st.toggle("Use Local Mistral (instead of OpenAI)")
    
    if not use_local_mistral:
        openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        st.markdown("[Get your OpenAI API key](https://platform.openai.com/account/api-keys)")
    else:
        st.info("Using local Mistral model via Ollama")
        st.markdown("""
        Ensure Ollama is running with:
        ```bash
        ollama serve
        ```
        """)

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.text_chunks = None
    st.session_state.vectorstore = None

def process_pdf():
    with st.spinner("Processing PDF..."):
        # Read PDF text
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(text)
        st.session_state.text_chunks = text_chunks
        
        # Create embeddings
        if use_local_mistral:
            embeddings = OllamaEmbeddings(model="mistral:7b-instruct-q4_0")
        else:
            from langchain.embeddings.openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        vectorstore = FAISS.from_texts(text_chunks, embeddings)
        st.session_state.vectorstore = vectorstore
        st.session_state.processed = True
    st.success("PDF processed! You can now ask questions.")

if uploaded_file is not None and not st.session_state.processed:
    if not use_local_mistral and not openai_api_key:
        st.warning("Please enter your OpenAI API key!")
    else:
        process_pdf()

# Question input
question = st.text_input("Ask a question about the PDF:")

def get_mistral_response(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Use the following context to answer the question at the end.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    response = requests.post(
        "http://localhost:11435/api/generate",
        json={
            "model": "mistral:7b-instruct-q4_0",
            "prompt": prompt,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        return json.loads(response.text).get("response", "No answer found")
    else:
        raise Exception(f"Ollama API error: {response.text}")

if question and st.session_state.processed:
    with st.spinner("Thinking..."):
        docs = st.session_state.vectorstore.similarity_search(question)
        
        if use_local_mistral:
            try:
                answer = get_mistral_response(question, docs)
                st.write(f"**Answer:** {answer}")
                st.write("⚡ Powered by local Mistral 7B")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            from langchain.llms import OpenAI
            llm = OpenAI(openai_api_key=openai_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=question)
            st.write(f"**Answer:** {response}")
            st.write("🌐 Powered by OpenAI")

# Add instructions at the bottom
with st.expander("Setup Instructions"):
    st.markdown("""
    **For Local Mistral Setup:**
    1. Install Ollama: [ollama.com](https://ollama.com)
    2. Pull the model:
    ```bash
    ollama pull mistral:7b-instruct-q4_0
    ```
    3. Run the server:
    ```bash
    ollama serve
    ```
    """)