from fpdf import FPDF
from PyPDF2 import PdfMerger
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings   ######## for embedding (instead of ollama embedding)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()






os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

prompt = ChatPromptTemplate.from_template(
    """
You are an experienced Network Operations Center (NOC) engineer. Generate a detailed incident report based on the provided input. Include the following:

A clear and concise summary of the issue.
Detailed steps taken to diagnose and resolve the issue.
Key metrics or data relevant to the incident.
The root cause of the problem.
The resolution and any follow-up actions required.
Additional recommendations to prevent recurrence.
Ensure the documentation is professional, thorough, and suitable for technical review.
<context>
{context}
<context>
Question:{input}
"""
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("dataset")   ############# data ingestion
        st.session_state.docs = st.session_state.loader.load()      ######## document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input("Enter your query from the dataset")
if st.button("Document embedding"):
    create_vector_embeddings()
    st.write("Vector database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time is: {time.process_time() - start}") 
    st.write(response['answer'])

    ### Streamlit  
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------------------------")

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)

        def add_content(self, text):
            self.add_page()
            self.set_font("Arial", size=12)
            self.multi_cell(0, 10, text)

    def append_to_pdf(existing_pdf, new_content):
        # Create a temporary PDF with the new content
        temp_pdf = "temp_append.pdf"
        pdf = PDF()
        pdf.add_content(new_content)
        pdf.output(temp_pdf)

        # Merge the original and the new PDF
        merger = PdfMerger()
        merger.append(existing_pdf)
        merger.append(temp_pdf)
        merger.write(existing_pdf)
        merger.close()

        print(f"Appended content saved to: {existing_pdf}")

    # Append the new response to the main PDF
    append_to_pdf("initial.pdf", response['answer'])
