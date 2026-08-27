from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from typing import List
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
import requests
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()  # Load environment variables from .env file

# search tool
search_tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")


@tool
def get_weather(city: str) -> str:
    """Ambil informasi cuaca terkini untuk sebuah kota menggunakan Weatherstack API.

    Args:
        city: Nama kota, misalnya 'Jakarta' atau 'Bandung'
    """
    url = "http://api.weatherstack.com/current"
    params = {
        "access_key": os.getenv("WEATHERSTACK_API_KEY"),
        "query": city,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "error" in data:
        return f"Error: {data['error'].get('info', 'Gagal mengambil data cuaca')}"

    location = data["location"]["name"]
    country = data["location"]["country"]
    temp = data["current"]["temperature"]
    desc = data["current"]["weather_descriptions"][0]
    humidity = data["current"]["humidity"]

    return (
        f"Cuaca di {location}, {country}: {desc}, "
        f"suhu {temp}°C, kelembapan {humidity}%"
    )


tools = [search_tool, get_weather]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

llm_with_tools = llm.bind_tools(tools)


# Define the state structure for the chatbot
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define the chatbot node function
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# Initialize the graph and checkpoint
conn = sqlite3.connect("chatbot_state.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)
graph = StateGraph(ChatState)

# add nodes to the graph
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# add edges to the graph
graph.add_edge(START, "chat_node")
# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

# Compile the graph
chat_bot = graph.compile(checkpointer=checkpoint)


# Function to retrieve all unique thread IDs from the checkpoint
def get_all_thread():
    all_thread = set()
    for ckpt in checkpoint.list(None):
        all_thread.add(ckpt.config["configurable"]["thread_id"])

    return list(all_thread)
