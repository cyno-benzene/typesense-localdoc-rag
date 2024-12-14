# langchain modules for PDF documents
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# langchain modules for chat
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# langchain modules for Google AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# python typing and i/o
from typing import List, Dict
import os

#Typesense 
import typesense
from langchain_community.vectorstores import Typesense
from config import PDF_DIRECTORY, TYPESENSE_API_KEY


class ChatBot:
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro"
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001'
        )
        
        node = {
            "host": "localhost",  # For Typesense Cloud use xxx.a1.typesense.net
            "port": "8108",       # For Typesense Cloud use 443
            "protocol": "http"    # For Typesense Cloud use https
        }
        self.typesense_collection_name = "lang-chain"
        
        # Setup Typesense client
        self.typesense_client = typesense.Client(
            {
            "nodes": [node],
            "api_key": TYPESENSE_API_KEY,
            "connection_timeout_seconds": 5
            }
        )

        self.init_vectorstore()

    def init_vectorstore(self):
        """
        Set up vector store, avoiding reindexing if collection exists and has documents
        """
        try:
            # Check if collection exists
            collection_exists = False
            try:
                collection_info = self.typesense_client.collections[self.typesense_collection_name].retrieve()
                collection_exists = True
            except Exception as e:
                # Collection does not exist
                collection_exists = False
            
            # If collection doesn't exist or is empty, create and populate it
            if not collection_exists or collection_info.get('num_documents', 0) == 0:
                default_docs = self.get_docs()
                self.vectorstore = Typesense.from_documents(
                    documents=default_docs,  # Start with saved documents
                    embedding=self.embeddings,
                    typesense_client=self.typesense_client,
                    typesense_collection_name=self.typesense_collection_name 
                )
            else:
                # Collection exists and has documents, create vectorstore without reindexing
                self.vectorstore = Typesense(
                    typesense_client=self.typesense_client,
                    embedding=self.embeddings,
                    typesense_collection_name=self.typesense_collection_name
                )
        
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            raise

    def load_pdf_dir(self):
        """loads all the pdfs in the a directory and returns the list"""
        documents = []
        for file in os.listdir(PDF_DIRECTORY):
            pdf_path = os.path.join(PDF_DIRECTORY, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        return documents

    def get_docs(self):
        """return chunk (split) docs, uses CharacterTextSplitter with chunk size = 1000 and overlap = 0 """
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = self.load_pdf_dir()
        docs = text_splitter.split_documents(documents)
        return docs

    def generate_response(self, query: str, chat_history: List[Dict]):
        """ returns the retriever"""
        relevant_docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in relevant_docs
        ])

        prompt = ChatPromptTemplate.from_template("""
            Provide a comprehensive answer using the following context and chat history. Include citations to source documents.
            Context Documents:
            {context}

            Chat History:
            {chat_history}

            User Query: {query}
        """)

        chain = (
            {
                "context": lambda x: context,
                "chat_history": lambda x: chat_history,
                "query": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke(query)

        return response, relevant_docs


