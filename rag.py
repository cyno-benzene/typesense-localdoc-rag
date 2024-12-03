import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Typesense
from config import PDF_DIRECTORY, TYPESENSE_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_pdf_dir():
    """loads all the pdfs in the a directory and returns the list"""
    documents = []
    for file in os.listdir(PDF_DIRECTORY):
        pdf_path = os.path.join(PDF_DIRECTORY, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents

def list_docs():
    """List all the documents in the directory"""
    doc_titles = []

    for file in os.listdir(PDF_DIRECTORY):
        title = os.path.splitext(file)[0]
        doc_titles.append(title)

    return doc_titles

def get_docs():
    """return chunk (split) docs, uses CharacterTextSplitter with chunk size = 1000 and overlap = 0 """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = load_pdf_dir()
    docs = text_splitter.split_documents(documents)
    return docs

def get_embeddings():
    """Vector embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

def get_docsearch():
    """retriever using typesense"""
    docs = get_docs()
    embeddings = get_embeddings()
    docsearch = Typesense.from_documents(
        docs,
        embeddings,
        typesense_client_params={
            "host": "localhost",  # Use xxx.a1.typesense.net for Typesense Cloud
            "port": "8108",  # Use 443 for Typesense Cloud
            "protocol": "http",  # Use https for Typesense Cloud
            "typesense_api_key": TYPESENSE_API_KEY,
            "typesense_collection_name": "lang-chain",
        },
    )
    return docsearch

def get_retriever():
    """ returns the retriever"""
    docsearch = get_docsearch()
    retriever = docsearch.as_retriever()
    return retriever




