import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
import requests
import json
import os
from contextlib import contextmanager
import time
from google.generativeai import configure, GenerativeModel
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.callbacks import get_openai_callback

# Set page title and configure page
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1.5rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .api-info {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    .chat-container {
        margin-top: 2rem;
        border-top: 1px solid #eee;
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.text_chunks = None
    st.session_state.vectorstore = None
    st.session_state.api_cost = 0
    st.session_state.api_tokens_used = 0
    st.session_state.model_name = None
    st.session_state.last_model_type = None
    st.session_state.pdf_content = None
    st.session_state.api_calls = 0
    st.session_state.rate_limit_remaining = None
    st.session_state.current_model_type = None
    st.session_state.rerun_flag = False  # Add a flag to control reruns
    st.session_state.current_file_name = None  # Add file name tracking

# Main title
st.title("📄 PDF Q&A Assistant")
st.markdown("### Ask questions about your PDF documents using AI")

# Function to update API info display
def update_api_info(api_info_container):
    with api_info_container:
        st.empty()  # Clear previous content
        
        if st.session_state.current_model_type == "OpenAI":
            st.markdown("#### OpenAI API Usage")
            st.markdown(f"**Model:** {st.session_state.model_name}")
            st.markdown(f"**API calls:** {st.session_state.api_calls}")
            st.markdown(f"**Estimated tokens used:** {int(st.session_state.api_tokens_used):,}")
            st.markdown(f"**Estimated cost:** ${st.session_state.api_cost:.5f}")
        
        elif st.session_state.current_model_type == "Google Gemini":
            st.markdown("#### Gemini API Usage")
            st.markdown(f"**Model:** {st.session_state.model_name}")
            st.markdown(f"**API calls made:** {st.session_state.api_calls}")
            st.markdown(f"**Rate limit status:** {st.session_state.rate_limit_remaining}")
            
# Function to process PDF with selected model
def process_pdf(text, openai_api_key=None, gemini_api_key=None, openai_model=None, 
                gemini_model=None, embedding_model=None, api_info_container=None):
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    st.session_state.text_chunks = text_chunks
    
    # Create embeddings based on selected model
    if st.session_state.current_model_type == "OpenAI":
        if not openai_api_key:
            st.error("Please enter your OpenAI API key!")
            return False
        
        # Track token usage
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        try:
            embeddings = OpenAIEmbeddings(model=embedding_model)
            vectorstore = FAISS.from_texts(text_chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            
            # Estimate cost (rough estimation)
            total_chars = sum(len(chunk) for chunk in text_chunks)
            est_tokens = total_chars / 4  # Rough estimate
            st.session_state.api_tokens_used = est_tokens
            st.session_state.api_cost = (est_tokens / 1000) * 0.0001  # $0.0001 per 1K tokens for ada embeddings
            st.session_state.api_calls = len(text_chunks)
            
            # Update API info
            if api_info_container:
                update_api_info(api_info_container)
            return True
        except Exception as e:
            st.error(f"Error with OpenAI embeddings: {str(e)}")
            return False
    
    elif st.session_state.current_model_type == "Google Gemini":
        if not gemini_api_key:
            st.error("Please enter your Google Gemini API key!")
            return False
        
        try:
            # Configure Gemini API - FIXED: removed timeout option
            configure(api_key=gemini_api_key)
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=gemini_api_key
            )
            vectorstore = FAISS.from_texts(text_chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            
            # Estimate usage (Gemini doesn't provide clear cost metrics)
            st.session_state.api_calls = len(text_chunks)
            
            # Check rate limit
            try:
                genai.get_model(gemini_model)
                st.session_state.rate_limit_remaining = "Available"
            except Exception as e:
                if "quota" in str(e).lower():
                    st.session_state.rate_limit_remaining = "Quota exceeded"
                else:
                    st.session_state.rate_limit_remaining = "Unknown"
            
            # Update API info
            if api_info_container:
                update_api_info(api_info_container)
            return True
        except Exception as e:
            st.error(f"Error with Gemini embeddings: {str(e)}")
            st.error(f"Detail: {str(e)}")
            return False
    
    else:  # Local Mistral
        try:
            embeddings = OllamaEmbeddings(model="mistral:7b-instruct-q4_0")
            vectorstore = FAISS.from_texts(text_chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            return True
        except Exception as e:
            st.error(f"Error with local Mistral: {str(e)}")
            st.error("Make sure Ollama is running with 'ollama serve'")
            return False
        
# Sidebar for model selection and API keys
with st.sidebar:
    st.header("🛠️ Settings")
    
    # Model selection
    model_type = st.selectbox(
        "Select AI Provider:",
        ["OpenAI", "Google Gemini", "Local Mistral"],
        key="model_select"
    )
    
    # Store current model type in session state
    st.session_state.current_model_type = model_type
    
    # Model-specific settings
    if model_type == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key:", type="password")
        openai_model = st.selectbox(
            "Select OpenAI Model:",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            key="openai_model_select"
        )
        st.session_state.model_name = openai_model
        embedding_model = "text-embedding-ada-002"
        
        if not openai_api_key:
            st.markdown("[Get your OpenAI API key](https://platform.openai.com/account/api-keys)")
    
    elif model_type == "Google Gemini":
        gemini_api_key = st.text_input("Google Gemini API Key:", type="password")
        gemini_model = st.selectbox(
                            "Select Gemini Model:",
                            [
                                "models/gemini-2.0-flash",
                                "models/gemini-2.0-pro-exp",
                                "models/gemini-2.0-flash-lite"
                            ],
                            key="gemini_model_select"
)
        st.session_state.model_name = gemini_model
        embedding_model = "models/embedding-001"
        
        if not gemini_api_key:
            st.markdown("[Get your Gemini API key](https://ai.google.dev/)")
    
    else:  # Local Mistral
        st.info("Using local Mistral model via Ollama")
        st.session_state.model_name = "mistral:7b-instruct-q4_0"
        st.markdown("""
        Ensure Ollama is running with:
        ```bash
        ollama serve
        ```
        """)
        # Set these to None for Local Mistral option
        openai_api_key = None
        gemini_api_key = None
        openai_model = None
        gemini_model = None
        embedding_model = None
    
    st.markdown("---")
    st.header("📊 API Usage")
    
    # Display API usage information
    api_info_container = st.container()
    
    with api_info_container:
        # Will be populated after API calls
        if model_type != "Local Mistral":
            st.markdown("*API usage will appear here after processing*")

# Check if model type changed - set flag instead of immediate rerun
if "last_model_type" in st.session_state and st.session_state.last_model_type != st.session_state.current_model_type:
    st.session_state.processed = False
    st.session_state.pdf_content = None
    st.session_state.rerun_flag = True  # Flag for rerun instead of immediate rerun

# Update last model type
if "last_model_type" in st.session_state and st.session_state.last_model_type != st.session_state.current_model_type:
    st.session_state.last_model_type = st.session_state.current_model_type

# Handle rerun at end of script to avoid loops
rerun_needed = False

# Main content area - PDF Upload and Processing
st.subheader("1️⃣ Upload Your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

# Check if file changed to reset state
if uploaded_file is not None:
    current_file_name = uploaded_file.name
    if "current_file_name" not in st.session_state or st.session_state.current_file_name != current_file_name:
        st.session_state.current_file_name = current_file_name
        st.session_state.processed = False
        st.session_state.pdf_content = None
        rerun_needed = True

# Process the PDF
if uploaded_file is not None:
    # Read PDF text if not already done or if model changed
    if not st.session_state.pdf_content:
        with st.spinner("Reading PDF..."):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Store PDF content to avoid re-processing
            st.session_state.pdf_content = text
            st.success("PDF loaded successfully!")
    
    # Button to process with selected model
    if not st.session_state.processed:
        process_button = st.button("Process PDF with Selected Model")
        if process_button:
            with st.spinner("Processing PDF with AI..."):
                success = process_pdf(
                    st.session_state.pdf_content,
                    openai_api_key=openai_api_key if 'openai_api_key' in locals() else None,
                    gemini_api_key=gemini_api_key if 'gemini_api_key' in locals() else None,
                    openai_model=openai_model if 'openai_model' in locals() else None,
                    gemini_model=gemini_model if 'gemini_model' in locals() else None,
                    embedding_model=embedding_model if 'embedding_model' in locals() else None,
                    api_info_container=api_info_container
                )
                if success:
                    st.session_state.processed = True
                    st.success("PDF processed! You can now ask questions.")
                    rerun_needed = True
    else:
        # Offer to reprocess with current model
        if st.button("Reprocess PDF with Selected Model"):
            st.session_state.processed = False
            rerun_needed = True
else:
    # Reset state when no file is uploaded
    if st.session_state.pdf_content is not None:
        st.session_state.pdf_content = None
        st.session_state.processed = False
        st.session_state.current_file_name = None
        st.session_state.rerun_flag = True

# Question and answer section - Only appears when a document is processed
if uploaded_file is not None and st.session_state.processed:
    st.markdown("---")
    st.subheader("2️⃣ Ask Questions About Your PDF")
    
    with st.container():
        question = st.text_input("Enter your question:", key="question_input")
        
        if question:
            with st.spinner("Analyzing document..."):
                docs = st.session_state.vectorstore.similarity_search(question)
                
                try:
                    # OpenAI
                    if st.session_state.current_model_type == "OpenAI":
                        llm = ChatOpenAI(
                            api_key=openai_api_key,
                            model_name=openai_model,
                            temperature=0
                        )
                        chain = load_qa_chain(llm, chain_type="stuff")
                        
                        with get_openai_callback() as cb:
                            response = chain.run(input_documents=docs, question=question)
                            st.markdown("### Answer")
                            st.write(response)
                            st.markdown("---")
                            st.markdown(f"*Powered by {openai_model}*")
                            
                            # Update API info
                            st.session_state.api_cost += cb.total_cost
                            st.session_state.api_tokens_used += cb.total_tokens
                            st.session_state.api_calls += 1
                            update_api_info(api_info_container)
                    
                    # Gemini
                    elif st.session_state.current_model_type == "Google Gemini":
                        configure(api_key=gemini_api_key)
                        context = "\n".join([doc.page_content for doc in docs])
                        prompt = f"""
                        Use the following context to answer the question.
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        
                        Answer:
                        """
                        
                        model = GenerativeModel(gemini_model)
                        response = model.generate_content(prompt)
                        
                        st.markdown("### Answer")
                        st.write(response.text)
                        st.markdown("---")
                        st.markdown(f"*Powered by {gemini_model}*")
                        
                        # Update API info
                        st.session_state.api_calls += 1
                        update_api_info(api_info_container)
                    
                    # Local Mistral
                    else:
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
                            answer = json.loads(response.text).get("response", "No answer found")
                            st.markdown("### Answer")
                            st.write(answer)
                            st.markdown("---")
                            st.markdown("*Powered by local Mistral 7B*")
                        else:
                            st.error(f"Ollama API error: {response.text}")
                
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
                    if st.session_state.current_model_type == "Local Mistral":
                        st.error("Make sure Ollama is running with 'ollama serve'")

# Setup instructions
with st.expander("📚 Setup Instructions"):
    st.markdown("""
    ### Setting Up API Keys
    
    #### OpenAI
    1. Create an account at [OpenAI](https://platform.openai.com/signup)
    2. Generate an API key at [API Keys](https://platform.openai.com/account/api-keys)
    3. Paste the key in the sidebar
    
    #### Google Gemini
    1. Create an account at [Google AI Studio](https://ai.google.dev/)
    2. Get your API key from [API Keys](https://console.cloud.google.com/apis/credentials)
    3. Paste the key in the sidebar
    
    #### Local Mistral
    1. Install Ollama: [ollama.com](https://ollama.com)
    2. Pull the model:
    ```bash
    ollama pull mistral:7b-instruct-q4_0
    ```
    3. Run the server:
    ```bash
    ollama serve
    ```
    
    ### Required Packages
    ```bash
    pip install streamlit PyPDF2 langchain langchain-community
    pip install faiss-cpu openai
    pip install google-generativeai langchain-google-genai
    pip install requests
    ```
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain")

# Handle controlled reruns at the very end of the script
if st.session_state.rerun_flag:
    st.session_state.rerun_flag = False  # Reset flag first
    st.rerun()
elif rerun_needed:
    st.rerun()