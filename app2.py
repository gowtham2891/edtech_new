import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
import tempfile

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following questions based only on the provided context.
    Think step by step before providing a detailed answer.
    I will tip you $1000 if user finds the answer helpful.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

def load_docs(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    text_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(text_documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_documents(final_docs, embeddings)
    
    os.unlink(tmp_path)
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
        
        # Clear the current question
        st.session_state.current_question = ""

def handle_input():
    if st.session_state.user_input:
        st.session_state.current_question = st.session_state.user_input
        st.session_state.user_input = ""  # Clear the input
        process_question()

def main():
    st.set_page_config(
        page_title="📚 Document Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stChat {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .assistant-message {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload your PDF document",
            type=['pdf'],
            help="Upload a PDF file to start chatting about its contents"
        )
        
        if uploaded_file and (not st.session_state.vectors or 
            uploaded_file.name != getattr(st.session_state, 'last_file_name', None)):
            with st.spinner("Processing document... Please wait"):
                vectors, chunks = load_docs(uploaded_file)
                st.session_state.vectors = vectors
                st.session_state.chunks = chunks
                st.session_state.last_file_name = uploaded_file.name
                st.success("✅ Document processed successfully!")

    # Main chat interface
    st.title("📚 Document Chat Assistant")
    
    if not st.session_state.vectors:
        st.info("👈 Please upload a PDF document to start chatting")
        st.markdown("""
        ### How to use:
        1. Upload your PDF document using the sidebar
        2. Wait for the document to be processed
        3. Start asking questions about your document
        4. The AI will provide answers based on the document content
        """)
    else:
        # Chat interface
        st.markdown("### Chat with your document")
        
        # Display chat history
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
        
        # User input
        st.text_input(
            "Ask a question about your document:",
            key="user_input",
            on_change=handle_input
        )
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []

if __name__ == "__main__":
    main()