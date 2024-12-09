import streamlit as st
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

import tempfile

# Load environment variables
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Configuration
CLASSES = ["8", "9", "10", "11", "12"]
SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology"]
MAX_CHAPTERS = 15

# Initialize session state
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'pdf_paths' not in st.session_state:
    st.session_state.pdf_paths = None

class ContentManager:
    def __init__(self):
        # Increased chunk size for more context
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def get_relevant_pdfs(self, selected_class: str, selected_subject: str, selected_chapter: int) -> List[str]:
        """
        Determine which PDFs should be loaded based on selection criteria
        """
        pdf_paths = []
        
        # Get PDFs from previous class (if applicable)
        if int(selected_class) > 8:
            prev_class = str(int(selected_class) - 1)
            for chapter in range(1, MAX_CHAPTERS + 1):
                pdf_path = f"content/class_{prev_class}/{selected_subject}/chapter_{chapter}.pdf"
                if os.path.exists(pdf_path):
                    pdf_paths.append(pdf_path)
        
        # Get PDFs from current class up to selected chapter
        for chapter in range(1, selected_chapter + 1):
            pdf_path = f"content/class_{selected_class}/{selected_subject}/chapter_{chapter}.pdf"
            if os.path.exists(pdf_path):
                pdf_paths.append(pdf_path)
                
        return pdf_paths
    
    def load_pdfs(self, pdf_paths: List[str]) -> tuple:
        """
        Load multiple PDFs and process them
        """
        all_docs = []
        
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                st.warning(f"Error loading {pdf_path}: {str(e)}")
                continue
        
        if not all_docs:
            return None, None
        
        final_docs = self.text_splitter.split_documents(all_docs)
        
        persist_directory = 'db'
        vectors = Chroma.from_documents(
            documents=final_docs,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        return vectors, final_docs

def setup_retrieval_chain(vectors, chunks, class_level, subject):
    """
    Set up the retrieval chain with enhanced prompt and configuration
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,  # Slightly increased for more detailed responses
            # max_tokens=2000  # Increased token limit for longer responses
        )
        
        # Enhanced prompt template with specific instructions for detailed responses
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert educational assistant specializing in {subject} for Class {class_level} students.
            Your task is to provide a comprehensive and detailed response to the following question,
            using only the information from the provided context.

            Guidelines for your response:
            1. Break down complex concepts into clear, understandable parts
            2. Include relevant examples and illustrations where appropriate
            3. Use proper terminology while explaining in student-friendly language
            4. If the question involves calculations, show all steps clearly
            5. For theoretical concepts, provide real-world applications
            6. Ensure completeness while maintaining clarity

            Context from educational materials:
            {context}

            Question: {input}

            Response structure:
            1. Main concept explanation
            2. Detailed breakdown and analysis
            3. Examples or applications
            4. Summary of key points
            
            Please provide your detailed response:
            """
        )
        
        # Create document chain with enhanced configuration
        document_chain = create_stuff_documents_chain(
            llm, 
            prompt,
            document_variable_name="context",
        )
        
        # Enhanced retriever configuration
        vectorstore_retriever = vectors.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,  # Increased for better coverage
                "fetch_k": 10,  # Fetch more documents initially
                "score_threshold": 0.5  # Only include relevant matches
            }
        )
        
        keyword_retriever = BM25Retriever.from_documents(
            chunks,
            k=6  # Increased for better coverage
        )
        
        # Create ensemble retriever with adjusted weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, keyword_retriever],
            weights=[0.8, 0.2]  # Adjusted weights to favor semantic search
        )
        
        retrieval_chain = create_retrieval_chain(
            retriever=ensemble_retriever,
            combine_documents_chain=document_chain
        )
        
        def enhanced_retrieval_chain(input_data):
            enhanced_input = {
                "input": input_data["input"],
                "class_level": class_level,
                "subject": subject
            }
            return retrieval_chain.invoke(enhanced_input)
        
        return enhanced_retrieval_chain
    
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Educational RAG System", layout="wide")
    st.header("📚 Educational Content Query System")
    
    content_manager = ContentManager()
    
    with st.form("content_selection"):
        st.subheader("Select Content")
        selected_class = st.selectbox("Class", CLASSES)
        selected_subject = st.selectbox("Subject", SUBJECTS)
        selected_chapter = st.number_input("Chapter", min_value=1, max_value=MAX_CHAPTERS, value=1)
        
        load_button = st.form_submit_button("Load Content")
        
        if load_button:
            with st.spinner('Loading relevant content...'):
                pdf_paths = content_manager.get_relevant_pdfs(
                    selected_class, 
                    selected_subject, 
                    selected_chapter
                )
                
                if not pdf_paths:
                    st.warning("No content found for the selected criteria.")
                    st.session_state.vectors = None
                    st.session_state.chunks = None
                    st.session_state.chain = None
                    st.session_state.pdf_paths = None
                else:
                    vectors, chunks = content_manager.load_pdfs(pdf_paths)
                    
                    if vectors and chunks:
                        st.session_state.vectors = vectors
                        st.session_state.chunks = chunks
                        st.session_state.pdf_paths = pdf_paths
                        
                        st.session_state.chain = setup_retrieval_chain(
                            vectors, 
                            chunks, 
                            selected_class, 
                            selected_subject
                        )
                        
                        st.success(f"Successfully loaded content from {len(pdf_paths)} PDFs!")
    
    if st.session_state.chain:
        st.divider()
        user_question = st.text_input("Ask a question about the content:")
        
        if user_question:
            with st.spinner('Generating detailed response...'):
                try:
                    response = st.session_state.chain({
                        "input": user_question,
                    })
                    result = response["answer"]
                    
                    st.markdown("### Detailed Answer")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()