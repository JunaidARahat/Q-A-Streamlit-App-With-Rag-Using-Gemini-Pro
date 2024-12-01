import streamlit as st
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

# Streamlit page configuration for wider layout and a custom favicon
st.set_page_config(page_title="Gemini Pro Question-Answer App", layout="wide", page_icon="🤖")

# Adding custom styles for an advanced layout
st.markdown(
    """
    <style>
    /* Global Styles */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f9;
        color: #333;
    }
    
    /* Title Styling */
    .stTitle {
        text-align: center;
        color: #2D3748;
        font-size: 42px;
        font-weight: bold;
        margin-top: 30px;
    }

    .stHeader {
        text-align: center;
        font-size: 26px;
        color: #4A90E2;
        margin-top: 10px;
    }

    /* Input Field Styling */
    .stTextInput textarea {
        border-radius: 8px;
        padding: 10px;
        font-size: 18px;
        border: 2px solid #ddd;
        background-color: #fff;
        transition: border 0.3s ease;
    }
    
    .stTextInput textarea:focus {
        border-color: #4A90E2;
    }
    
    /* Answer Styling */
    .answer-card {
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 30px;
        font-size: 18px;
        color: #333;
    }

    /* Button Styles */
    .stButton > button {
        background-color: #4A90E2;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #357ABD;
    }

    /* Placeholder for info section */
    .stInfo {
        background-color: #e0f7fa;
        padding: 10px;
        font-size: 16px;
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown("<h1 class='stTitle'>🤖 Q&A RAG Application using Gemini Pro</h1>", unsafe_allow_html=True)

# File loading and processing
loader = PyPDFLoader("D:\\Question Answer App Rag Dec 2024\\Q-A-Streamlit-App-With-Rag-Using-Gemini-Pro\\DATA\\research_paper.pdf")
data = loader.load()

# Text splitting for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Embedding and retrieval setup
vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Input field for query
query = st.text_input("Ask me anything:", placeholder="Type your question here...", key="query_input")

# Define system prompt template  
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Set up the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Answer generation section
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    st.markdown(f"<div class='answer-card'><h3>🤔 **Answer:**</h3><p>{response['answer']}</p></div>", unsafe_allow_html=True)
else:
    st.markdown(
        "<div class='stInfo'>Please ask a question to get an answer!</div>",
        unsafe_allow_html=True
    )
