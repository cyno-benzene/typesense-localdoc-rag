# Local Document RAG QnA with Typesense + LangChain + Gemini + Chainlit  

This is a Conversational RAG Chatbot built using Chainlit, Langchain, Gemini for embeddings generation and LLM and Typesense as a vectorstore. It allows for coversation over saved documents or documents uploaded during conversation. 



## Pre-requisites
1. Python 3.9+
2. Docker for locally hosting the Typesense server using the compose.yml file in the repository
3. Gemini(GoogleGenerativeAI) API key
QnA with spontanously uploaded PDF is yet to be implemented.

## Local Setup
1. Clone the project
2. Install the dependencies using the given `requirements.txt` file
    ```
    pip install -r requirement.txt
    ```
3. Run the local install of Typesense server using the `compose.yml` config in this repository(Requires docker installation on the system). Use the following command. 
    ```
    docker compose up
    ```
4. Copy the .env.example file and create a `.env` file at the root of the project.
5. Set the values of required environment variables in the .env file that was created.
6. Replace the documents in the `downloaded_pdfs` directory with the ones you want to have a conversation with.
7. Run the following command to start the Chainlit application. 
    ```
    chainlit run main.py -w
    ```
On successful run, the app opens automatically in your default browser on `localhost:8000`

## References
- <a href="https://typesense.org/">Typesense</a>
- <a href="https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.typesense.Typesense.html">Langchain methods for Typesense as vector store</a>
- <a href="https://python.langchain.com/docs/tutorials/rag/#jupyter-notebook">Langchain RAG</a>
- <a href="https://docs.chainlit.io/get-started/overview">Chainlit</a>

### Last Commit Updates
- Refactored code with newer LangChain methods for Typesense. 
- Created a Chatbot class 
- App now loads and refreshes faster with Chatbot class app instance. 