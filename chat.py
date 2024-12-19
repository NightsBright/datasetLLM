import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

## for langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A CHATBOT"

## template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,temperature,max_tockens):
    groq_api_key=os.getenv("GROQ_API_KEY")
    llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
    output_parser=StrOutputParser()

    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer


##### the app part
st.title("test v1")
st.sidebar.title("settings")
api_key = st.sidebar.text_input("enter your api key:",type="password")

temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tockens = st.sidebar.slider("Max Tockens",min_value=50,max_value=300,value=150)


## interface for user input

st.write("type in an error")
user_input=st.text_input("You: ")
if user_input:
    response=generate_response(user_input,temperature,max_tockens)
    st.write(response)
else:
    st.write("Please provide an error")