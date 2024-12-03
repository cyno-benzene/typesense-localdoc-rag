from rag import get_docsearch, get_retriever, list_docs
import chainlit as cl
from langchain.chains import LLMChain, RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from config import set_env
# from document_utils.extract import extract_text_from_pdf 
import os 

set_env()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=get_retriever()
)

@cl.on_chat_start
async def on_chat_start():
    # List existing documents
    existing_docs = list_docs()
    doc_list = "\n".join([doc for doc in existing_docs])
    
    await cl.Message(
        content=f"Existing documents:\n{doc_list}\n\nPlease upload a PDF file or start asking to begin!",
    ).send()



@cl.on_message
async def main(message: cl.Message):

    if not message.elements:
        await cl.Message(content="PDF Uploaded!").send()
    else:
        documents = [file for file in message.elements if "pdf" in file.mime]
        pdf_folder = "downloaded_pdfs"
        if not os.path.exists(pdf_folder):
            print("Creating PDF folder")
            os.makedirs(pdf_folder)

        for file in documents:
            file_path = os.path.join(pdf_folder, file.name)
            with open(file_path, "wb") as f:
                f.write(file.content)

        await cl.Message(
            content=f"Files uploaded and saved successfully.",
        ).send()
        # Process the PDF further 
        # this is approach is for testing purposess
    # message_history = ChatMessageHistory()


    query = message.content
    response = chain(query) 
    await cl.Message(
        content=f"{response['result']}",
    ).send()

