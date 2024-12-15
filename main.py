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
        author="AskHR"
    ).send()
    cl.user_session.set("disable_welcome", True)
    cl.user_session.set("chatbot", app)

@cl.on_message
async def main(message: cl.Message):
    chatbot = cl.user_session.get("chatbot")
    
    # Retrieve chat history
    history = cl.user_session.get("chat_history", [])
    
    response_msg = cl.Message(content="•••")
    await response_msg.send()

    await cl.sleep(2)

    dots = ["•", "••", "•••"]
    dot_index = 0
    
    # Show the loading animation while awaiting the response
    for _ in range(10):  # Loop for 10 seconds to show the animation
        response_msg.content = dots[dot_index]  # Update with the current dots
        await response_msg.update()
        dot_index = (dot_index + 1) % len(dots)  # Move to the next dot state
        await cl.sleep(0.3)  # Change the animation speed as needed


    stream = await chatbot.generate_response(message.content, history)
    response_msg.content = ""
    async for token in stream:
        await response_msg.stream_token(token + " ")
        await cl.sleep(0.05)  # Adjust the speed of token delivery

    await response_msg.update()

    # Update chat history
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=response_msg))
    cl.user_session.set("chat_history", history[-10:])  # Keep last 10 messages

