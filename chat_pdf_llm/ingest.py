from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader ,PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from constant import CHROMA_SETTINGS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

persist_directory = "db"
# Load the PDF files from the directory
def main():
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

        

main()

