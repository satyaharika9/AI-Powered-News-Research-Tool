import os
import streamlit as st
import pickle
import time
import cohere
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import Cohere


load_dotenv()  # take environment variables from .env.
# Retrieve the Cohere API key from the environment
cohere_api_key = os.getenv('COHERE_API_KEY')
llm = Cohere(cohere_api_key=cohere_api_key)        
        

st.title("News Research Tool 📈")
st.sidebar.title("News Articles URL's")

urls = []

for i in range(3):
   url = st.sidebar.text_input(f"URL {i+1}")
   urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vectorIndex_coherex"
# To show the progress bar
main_placeholder = st.empty()
# create embeddings and save it to FAISS index
embeddings = CohereEmbeddings(model="embed-english-v3.0",)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    # for showing that data is loading
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000, chunk_overlap = 200)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    chunks = splitter.split_documents(data) # for document type we can use split_documents.
    
    

    vectorIndex_cohere = FAISS.from_documents(chunks, embeddings)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    
    vectorIndex_cohere.save_local(file_path)
   
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        vectorIndex_cohere = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorIndex_cohere.as_retriever()

        
        qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)
        result = qa({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)