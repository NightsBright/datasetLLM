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
import pandas as pd
import numpy as np
import joblib
import time
# Load the trained model
model = joblib.load('model.pkl')

# Read the dataset from a local file
df = pd.read_csv('dataset.csv')

# Define the steps for each incident type
steps_lower = {
    "fire": "Evacuate immediately and call the fire department.",
    "spill": "Contain the spill and wear protective equipment.",
    "common": "Follow standard safety procedures."
}

# Streamlit page title
st.title("Incident Root Cause Analysis")

# Text inputs for user inputs
actions_taken = st.text_area("Actions Taken / Current Progress", "")
issues_faced = st.text_area("Issues Faced During Implementation", "")

# Button to generate output
if st.button("Analyze and Generate Output"):
    st.write("## Analysis Output")

    with open("analysis_output.txt", "a") as file:
        for i in range(1):
            # Access the row using iloc (index-based)
            row = df.iloc[i]

            # Access columns: 'Incident ID', 'Incident Type', and other necessary columns
            incident_id = row['Incident ID']
            incident_type = row['Incident Type']
            jitter = row['Jitter']
            jitter = int(jitter.replace("ms", ""))
            latency = row['Latency']
            latency = int(latency.replace("ms", ""))
            packet_loss = row['Packet Loss']
            packet_loss = int(packet_loss.replace("%", ""))
            signal_strength = row['Signal Strength']

            # Determine steps based on incident type
            if incident_type.lower() in steps_lower:
                step = steps_lower[incident_type.lower()]
            else:
                step = steps_lower["common"]

            # Prepare input features for prediction
            input_features = np.array([[jitter, latency, packet_loss, signal_strength]])

            # Convert input features to DataFrame with appropriate column names
            feature_names = ['Jitter', 'Latency', 'Packet Loss', 'Signal Strength']
            input_features_df = pd.DataFrame(input_features, columns=feature_names)

            # Make the prediction
            predicted_issue = model.predict(input_features_df)

            # Display the details for each incident
            st.write(f"### Incident ID: {incident_id}")
            st.write(f"- Incident Type: {incident_type}")
            st.write(f"- Root Cause: {predicted_issue[0]}")
            st.write(f"- Steps: {step}")
            # Print all columns dynamically
            st.write("#### Incident Details")
            for column_name, value in row.items():
                st.write(f"- {column_name}: {value}")

            st.write("---")
            

            # Append output to the text file
            with open('example.txt', 'w') as file:
                file.write(f"Incident ID: {incident_id}\n")
                file.write(f"Incident Type: {incident_type}\n")
                file.write(f"Root Cause: {predicted_issue[0]}\n")
                file.write(f"Steps: {step}\n")
                file.write("Incident Details:\n")
                for column_name, value in row.items():
                    file.write(f"- {column_name}: {value}\n")
                file.write("-" * 40 + "\n")

        # Append user inputs to the text file
                file.write(f"Actions Taken / Current Progress: {actions_taken}\n")
                file.write(f"Issues Faced During Implementation: {issues_faced}\n")
                file.write("=" * 40 + "\n")


#start of srihari's model

# from dotenv import load_dotenv
# load_dotenv()


# # Load environment variables
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")

# # LLM and Prompt setup
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
# prompt = ChatPromptTemplate.from_template("""
# You are an experienced Network Operations Center (NOC) engineer. Generate a detailed incident report based on the provided input. Include the following:

# A clear and concise summary of the issue.
# Detailed steps taken to diagnose and resolve the issue.
# Key metrics or data relevant to the incident.
# The root cause of the problem.
# The resolution and any follow-up actions required.
# Additional recommendations to prevent recurrence.
# Ensure the documentation is professional, thorough, and suitable for technical review.
# <context>
# {context}
# <context>
# Question:{input}
# """)

# # Function to create vector embeddings
# def create_vector_embeddings():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         st.session_state.loader = PyPDFDirectoryLoader("dataset")   # Data ingestion
#         st.session_state.docs = st.session_state.loader.load()      # Document loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# # Load user input from a file
# with open('example.txt', 'r') as file:
#     file_content = ''.join(file.readlines())

# user_prompt = file_content

# # Initialize session_state.vectors to prevent attribute errors
# if "vectors" not in st.session_state:
#     st.session_state.vectors = None

# # Button to generate embeddings
# if st.button("Document embedding"):
#     create_vector_embeddings()
#     st.write("Vector database is ready")

# if user_prompt:
#     # Ensure vectors are initialized before accessing
#     if st.session_state.vectors:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()  # Safe access
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': user_prompt})
#         print(f"Response time is: {time.process_time() - start}")
#         st.write(response['answer'])

#         ### Streamlit  
#         with st.expander("Document similarity search"):
#             for i, doc in enumerate(response['context']):
#                 st.write(doc.page_content)
#                 st.write("----------------------------------------------")

#         class PDF(FPDF):
#             def __init__(self):
#                 super().__init__()
#                 self.set_auto_page_break(auto=True, margin=15)

#             def add_content(self, text):
#                 self.add_page()
#                 self.set_font("Arial", size=12)
#                 self.multi_cell(0, 10, text)

#         def append_to_pdf(existing_pdf, new_content):
#             # Create a temporary PDF with the new content
#             temp_pdf = "temp_append.pdf"
#             pdf = PDF()
#             pdf.add_content(new_content)
#             pdf.output(temp_pdf)

#             # Merge the original and the new PDF
#             merger = PdfMerger()
#             merger.append(existing_pdf)
#             merger.append(temp_pdf)
#             merger.write(existing_pdf)
#             merger.close()

#             print(f"Appended content saved to: {existing_pdf}")

#         # Append the new response to the main PDF
#         append_to_pdf("initial.pdf", response['answer'])
#     else:
#         st.write("Please initialize the vector embeddings first.")
