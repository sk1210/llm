import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

import os



def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", show_progress=True)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title="Chat with multiple Papers",
                       page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask question about your doc : ")

    with st.sidebar:
        st.subheader("Input papers")
        pdf_files = st.file_uploader("Upload papers here ",  accept_multiple_files=True)
        if st.button("Index"):
            with st.spinner(" Processing ... "):
                raw_text = get_pdf_text(pdf_files)

                text_chunks = get_text_chunks(raw_text)

                vector_store = get_vectorstore(text_chunks)

if __name__ == '__main__':
    main()
