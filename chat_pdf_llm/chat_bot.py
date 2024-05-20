import streamlit as st
import os
import sys
import base64
import time
# import torch 
# from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
# from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 

from langchain.chains import RetrievalQA

from constant import CHROMA_SETTINGS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import retrieval_qa
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

persist_directory = "db"

st.set_page_config(layout="wide")

# persist_directory = "db"

@st.cache_resource
def data_ingestion():
    doc=r"C:\Langchain_chatbot\lanchain_practice\Langchain_\chat_pdf_llm\docs"
    for root,dirs,files in os.walk(doc):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader=PyPDFLoader(os.path.join(root,file))
        documents=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts=text_splitter.split_documents(documents)
        #create embeddings here
        print("Loading sentence transformers model")
        # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        #create vector store here
        # db=Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)        
        db=Chroma.from_documents(documents,OpenAIEmbeddings(),persist_directory=persist_directory)
        db.persist()
        db=None
    

def qa_llm():
    vector_db_name=r"C:\Langchain_chatbot\lanchain_practice\Langchain_\chat_pdf_llm"
    vector_db_in='db'
    persist_directory= os.path.join(vector_db_name,vector_db_in)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # db=Chroma.from_documents(persist_directory=persist_directory,embedding=embedding)
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages
# def display_conversation(history):
#     for i in range(len(history["generated"])):
#         message(history["past"][i], is_user=True, key=str(i) + "_user")
#         message(history["generated"][i],key=str(i))
def display_conversation(history):
    for i in range(len(history["generated"])):
        st.write(f"User: {history['past'][i]}")
        st.write(f"Bot: {history['generated'][i]}")

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/yadavparagyp'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        col1, col2= st.columns([1,2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)


            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]
                
            # Search the database for a response based on user input and update session state
            if user_input:
                answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)
        

        




if __name__ == "__main__":
    main()