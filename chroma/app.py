import streamlit as st
import os
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

# Create a prompt template
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
    """
    Load and process documents from uploaded PDF
    """
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        text_documents = loader.load()

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(text_documents)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Initialize Chroma with a persistent directory
        persist_directory = 'db'
        vectors = Chroma.from_documents(
            documents=final_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        return vectors, final_docs
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None, None

def setup_retrieval_chain(vectors, chunks):
    """
    Set up the retrieval chain with the LLM and retrievers
    """
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="Llama3-8b-8192"
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Set up retrievers
        vectorstore_retriever = vectors.as_retriever(   
            search_type="similarity",
            # search_kwargs={"k": 3}
        )
        keyword_retriever = BM25Retriever.from_documents(chunks)
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )
        
        return create_retrieval_chain(ensemble_retriever, document_chain)
    
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="RAG Application", layout="wide")
    st.header("📚 RAG Application")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    
    if uploaded_file:
        with st.spinner('Processing document...'):
            vectors, chunks = load_docs(uploaded_file)
            
            if vectors and chunks:
                st.success("Document processed successfully!")
                retrieval_chain = setup_retrieval_chain(vectors, chunks)
                
                if retrieval_chain:
                    # User input
                    user_question = st.text_input("Ask a question about the document:")
                    
                    if user_question:
                        with st.spinner('Generating response...'):
                            try:
                                response = retrieval_chain.invoke({"input": user_question})
                                result = response["answer"]
                                
                                # Display result in a nice format
                                st.markdown("### Answer")
                                st.markdown(result)
                                
                            except Exception as e:
                                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()