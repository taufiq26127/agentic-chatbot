from agentic_chatbot_backend import chat_bot
from langchain_core.messages import HumanMessage
import streamlit as st

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

# loading previous messages from session state
for messages in st.session_state["messages_history"]:
    with st.chat_message(messages["role"]):
        st.write(messages["content"])

st.title("Agentic Chatbot with LangGraph")

user_input = st.chat_input("You: ", key="user_input")

if user_input:
    st.session_state["messages_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    response = chat_bot.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config=CONFIG
    )

    ai_message = response["messages"][-1].content

    st.session_state["messages_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    with st.chat_message("assistant"):
        st.write(ai_message)
