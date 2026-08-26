from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from typing import List
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

# model llm local
llm = ChatOllama(model="llama3.2:3b")


# Define the state structure for the chatbot
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# Define the chatbot node function
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Initialize the graph and checkpoint
conn = sqlite3.connect("chatbot_state.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph_builder = StateGraph(ChatState)

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chat_node)

# Connect the start and end nodes to the chatbot node
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
chat_bot = graph_builder.compile(checkpointer=checkpoint)


def get_all_thread():
    all_thread = set()
    for ckpt in checkpoint.list(None):
        all_thread.add(ckpt.config["configurable"]["thread_id"])

    return list(all_thread)
