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
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
import tempfile

# Load environment variables
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Configuration
CLASSES = ["8", "9", "10", "11", "12"]
SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology"]
MAX_CHAPTERS = 15  # Adjust based on your curriculum

class ContentManager:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def get_relevant_pdfs(self, selected_class: str, selected_subject: str, selected_chapter: int) -> List[str]:
        """
        Determine which PDFs should be loaded based on selection criteria
        """
        pdf_paths = []
        
        # Get PDFs from previous class (if applicable)
        if int(selected_class) > 8:
            prev_class = str(int(selected_class) - 1)
            # Add all chapters from previous class of same subject
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
        
        # Split documents
        final_docs = self.text_splitter.split_documents(all_docs)
        
        # Initialize Chroma
        persist_directory = 'db'
        vectors = Chroma.from_documents(
            documents=final_docs,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        return vectors, final_docs

def setup_retrieval_chain(vectors, chunks, class_level, subject):
    """
    Set up the retrieval chain with the LLM and retrievers
    """
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="Llama3-8b-8192"
        )
        
        # Create prompt template with class and subject information
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context.
            This question is related to Class {class_level} {subject} curriculum.
            Think step by step before providing a detailed answer.
            
            <context>
            {context}
            </context>
            
            Question: {input}
            """
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm, 
            prompt
        )
        
        # Set up retrievers
        vectorstore_retriever = vectors.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Increased k for better coverage
        )
        keyword_retriever = BM25Retriever.from_documents(chunks)
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )
        
        retrieval_chain = create_retrieval_chain(ensemble_retriever, document_chain)
        
        # Return a function that includes the class and subject in the input
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
    
    # Initialize content manager
    content_manager = ContentManager()
    
    # Sidebar for selection
    with st.sidebar:
        st.subheader("Select Content")
        selected_class = st.selectbox("Class", CLASSES)
        selected_subject = st.selectbox("Subject", SUBJECTS)
        max_chapter = MAX_CHAPTERS  # You might want to make this dynamic based on class/subject
        selected_chapter = st.number_input("Chapter", min_value=1, max_value=max_chapter, value=1)
    
    # Main content area
    if selected_class and selected_subject and selected_chapter:
        with st.spinner('Loading relevant content...'):
            # Get relevant PDF paths
            pdf_paths = content_manager.get_relevant_pdfs(
                selected_class, 
                selected_subject, 
                selected_chapter
            )
            
            if not pdf_paths:
                st.warning("No content found for the selected criteria.")
                return
            
            # Load and process PDFs
            vectors, chunks = content_manager.load_pdfs(pdf_paths)
            
            if vectors and chunks:
                st.success(f"Successfully loaded content from {len(pdf_paths)} PDFs!")
                
                # Setup retrieval chain
                chain = setup_retrieval_chain(
                    vectors, 
                    chunks, 
                    selected_class, 
                    selected_subject
                )
                
                if chain:
                    # User input
                    user_question = st.text_input("Ask a question about the content:")
                    
                    if user_question:
                        with st.spinner('Generating response...'):
                            try:
                                response = chain({
                                    "input": user_question,
                                })
                                result = response["answer"]
                                
                                # Display result
                                st.markdown("### Answer")
                                st.markdown(result)
                                
                                # Display source information
                                # st.markdown("### Sources")
                                # for path in pdf_paths:
                                #     st.markdown(f"- {os.path.basename(path)}")
                                
                            except Exception as e:
                                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()