import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import tempfile
import time
import glob

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Enhanced styling
st.set_page_config(
    page_title="🌟 AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations and better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        font-size: 3rem !important;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* Chat container styling */
    .chat-container {
        padding: 1rem;
        border-radius: 15px;
        background: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #6B9FFF 0%, #4481EB 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        animation: slideInRight 0.5s ease-out;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e6e8 100%);
        color: #2c3e50;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        animation: slideInLeft 0.5s ease-out;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Input box styling */
    .stTextInput > div > div {
        border-radius: 25px !important;
        border: 2px solid #1E88E5 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px !important;
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        transition: transform 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1E88E5 0%, #1565C0 100%);
        color: white;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Enhanced prompt template
prompt = ChatPromptTemplate.from_template("""
    As a knowledgeable and friendly AI assistant, I'll help you understand the document.
    Let me analyze the context and provide a clear, detailed answer.
    
    Context:
    {context}
    
    Question: {input}
    
    Let me think about this step by step...
""")

def load_docs_from_directory(directory_path):
    """
    Load all PDFs from a specified directory
    """
    with st.spinner("🔍 Processing documents from directory... Please wait"):
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Get list of PDF files in directory
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        total_files = len(pdf_files)
        
        if total_files == 0:
            st.error("No PDF files found in the specified directory!")
            return None, None
        
        all_documents = []
        for i, pdf_path in enumerate(pdf_files):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                all_documents.extend(documents)
                
                # Update progress bar
                progress = (i + 1) / total_files
                progress_bar.progress(progress)
                st.write(f"Processed {os.path.basename(pdf_path)}")
                
            except Exception as e:
                st.warning(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
                continue
        
        if not all_documents:
            st.error("No documents were successfully processed!")
            return None, None
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(all_documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(final_docs, embeddings)
        
        st.success(f"✨ Successfully processed {total_files} PDF files!")
        return vectors, final_docs

def setup_chain(vectors, chunks):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192"
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    vectorstore_retriever = vectors.as_retriever()
    keyword_retriever = BM25Retriever.from_documents(chunks)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    return create_retrieval_chain(ensemble_retriever, document_chain)

def process_question():
    if st.session_state.current_question and st.session_state.vectors:
        with st.spinner("🤔 Thinking..."):
            retrieval_chain = setup_chain(st.session_state.vectors, st.session_state.chunks)
            response = retrieval_chain.invoke({"input": st.session_state.current_question})
            result = response["answer"]
            
            st.session_state.chat_history.append({
                "role": "user", 
                "content": st.session_state.current_question
            })
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": result
            })
            
            st.session_state.current_question = ""

def handle_input():
    if st.session_state.user_input:
        st.session_state.current_question = st.session_state.user_input
        st.session_state.user_input = ""
        process_question()

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 📚 Directory Selection")
        st.markdown("---")
        directory_path = st.text_input(
            "Enter the directory path containing PDFs:",
            help="Enter the full path to the directory containing your PDF files"
        )
        
        if directory_path and (not st.session_state.vectors or 
            directory_path != getattr(st.session_state, 'last_directory', None)):
            if os.path.isdir(directory_path):
                vectors, chunks = load_docs_from_directory("jesc1dd")
                if vectors and chunks:
                    st.session_state.vectors = vectors
                    st.session_state.chunks = chunks
                    st.session_state.last_directory = directory_path
                    
                    # Welcome message
                    if not st.session_state.chat_history:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"👋 Hello! I've processed all PDF files from the directory. Feel free to ask me any questions about them!"
                        })
            else:
                st.error("Invalid directory path!")

    # Main interface
    st.markdown("# 🌟 AI Document Assistant")
    
    if not st.session_state.vectors:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>👋 Welcome to AI Document Assistant!</h2>
            <p style='font-size: 1.2rem; color: #666;'>
                Let me help you understand your documents better. Here's how to get started:
            </p>
            <div style='margin: 2rem 0;'>
                <p>1. 📁 Enter the directory path containing your PDFs in the sidebar</p>
                <p>2. ⏳ Wait for processing (it's quick!)</p>
                <p>3. 💬 Start asking questions</p>
                <p>4. 🤖 Get intelligent responses</p>
            </div>
            <p style='color: #1E88E5;'>👈 Start by entering your directory path!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat interface
        st.markdown("### 💬 Chat with your documents")
        
        # Display chat history with enhanced styling
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">👤 You: {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="assistant-message">🤖 Assistant: {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
        
        # Input area with enhanced styling
        st.text_input(
            "💭 Ask me anything about your documents:",
            key="user_input",
            on_change=handle_input
        )
        
        # Control buttons
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()