import os
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from config import set_env
from rag import ChatBot 


set_env()
app = ChatBot()


@cl.on_chat_start
async def on_chat_start():
    
    await cl.Message(
        content=f"I can tell you anything about Python basics. You can start asking to begin!",
    ).send()

    cl.user_session.set("chatbot", app)


@cl.on_message
async def main(message: cl.Message):
    chatbot = cl.user_session.get("chatbot")
    
    # Retrieve chat history
    history = cl.user_session.get("chat_history", [])
    
    # Generate response
    response, docs = chatbot.generate_response(message.content, history)
    
    await cl.Message(content=response).send()
    
    # Update chat history
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=response))
    cl.user_session.set("chat_history", history[-10:])  # Keep last 10 messages

