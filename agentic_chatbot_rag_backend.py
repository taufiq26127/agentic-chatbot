from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import math
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from typing import Any

load_dotenv()


# LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)


def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH, embeddings=embeddings, allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    return retriever


# rag tool


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


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


# Tools

search_tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")

# Make tool list
tools = [search_tool, rag_tool, get_weather]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


# State
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


# Nodes 1
def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `get_weather` when the user asks about current weather for a location.\n\n"
            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [system_message, *state["messages"]]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# Nodes 2 - tool node
tool_node = ToolNode(tools)

# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


# Helper functions for Streamlit frontend
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])

    return list(all_threads)
