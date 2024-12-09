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



load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


prompt = ChatPromptTemplate.from_template(

    """
    Answer the folowing questions based only on the provided context.
    Think step by step before providing a detailed answer.
    I will tip you $1000 if user finds the answer helpful.
    <context>
    {context}
    </context>
    Question: {input}
    """

)




def load_docs(docs):

    loader = PyPDFLoader(docs)
    text_documents = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(text_documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_documents(final_docs, embeddings)
    return vectors, final_docs


    
def input(vectors, chunks):
    llm = ChatGroq(groq_api_key=groq_api_key,
               model_name = "Llama3-8b-8192")
    document_chain = create_stuff_documents_chain(llm, prompt)
    vectorstore_retreiver = vectors.as_retriever()
    keyword_retriever = BM25Retriever.from_documents(chunks)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.7, 0.3])

    retrieval_chain = create_retrieval_chain(ensemble_retriever, document_chain)
    
    return retrieval_chain



def main():

    st.set_page_config("RAG")
    st.header("RAG Application")
    vectors, chunks = load_docs("jesc101.pdf")
    retrieval_chain = input(vectors, chunks)

    user_question = st.text_input("Ask question from the uploaded PDFs")
    if user_question:
        response = retrieval_chain.invoke({"input": user_question})
        result = response["answer"]
        st.write(result)


if __name__ == "__main__":
    main()