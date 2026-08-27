from agentic_chatbot_rag_backend import chatbot, get_all_threads, ingest_rag_document

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

import streamlit as st
import uuid
import tempfile
import os
import datetime


# change list answer ai
def message_text(content):
    """Convert LangChain text or content blocks to renderable Markdown."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


def tool_name(message):
    """Get a readable tool name from a tool call or tool result."""
    return getattr(message, "name", None) or "tool"


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def render_tool_history(tools):
    if not tools:
        return
    with st.expander(f"🔧 {len(tools)} tool dipanggil", expanded=False):
        for name in tools:
            st.caption(f"• `{name}`")


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ========================= Global styling =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #eef3ff 0%, #f5f7fb 35%, #f7f9fc 100%);
    }

    /* Hide default streamlit chrome for a cleaner look */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #131a2b 0%, #1b2540 100%);
        border-right: 1px solid rgba(255,255,255,.05);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #dce6f7 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.04);
        color: #f5f8ff !important;
        text-align: left;
        border-radius: 12px;
        padding: .55rem .9rem;
        font-weight: 500;
        transition: all .18s ease;
        box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(120deg, rgba(76,110,245,.35), rgba(108,99,255,.35));
        border-color: #7fa4ff;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] .stButton > button:active {
        transform: scale(.98);
    }

    /* "New chat" primary button gets its own accent */
    [data-testid="stSidebar"] .stButton:first-of-type > button {
        background: linear-gradient(120deg, #4f6ef7, #7c6bff);
        border: none;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(79,110,247,.35);
    }
    [data-testid="stSidebar"] .stButton:first-of-type > button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,.08);
        margin: 1.1rem 0 .9rem 0;
    }

    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: inherit;
        padding: .9rem 1.2rem;
        font-size: .74rem;
        color: rgba(220,230,247,.45);
    }

    /* ---------- Hero header ---------- */
    .hero {
        position: relative;
        padding: 1.9rem 2rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(120deg, #2457d6 0%, #6c63ff 55%, #a06bff 100%);
        box-shadow: 0 18px 40px rgba(50, 82, 190, .28);
        margin-bottom: 1.6rem;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(255,255,255,.25), transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -.02em;
    }
    .hero p {
        margin: .5rem 0 0;
        opacity: .92;
        font-size: .96rem;
    }
    .hero-badges {
        margin-top: .9rem;
        display: flex;
        gap: .5rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.28);
        backdrop-filter: blur(4px);
        padding: .25rem .7rem;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 600;
    }

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 1rem 1.15rem;
        margin-bottom: .9rem;
        box-shadow: 0 4px 16px rgba(30, 45, 75, .07);
        animation: fadeInUp .28s ease;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] code {
        color: #172033 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #dceaff, #e8f0ff);
        border: 1px solid #c5d9fa;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #ffffff;
        border: 1px solid #e6e9f2;
    }
    [data-testid="stChatMessage"] pre {
        background: #eef2f8;
        color: #172033 !important;
        border-radius: 10px;
        padding: .8rem;
        border: 1px solid #e1e6ef;
    }

    /* ---------- Chat input ---------- */
    .stChatInputContainer, [data-testid="stChatInput"] {
        padding-bottom: 1rem;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
    }

    /* ---------- Empty state ---------- */
    .hint {
        text-align: center;
        color: #4a5670;
        padding: 3rem 1.5rem;
        background: rgba(255,255,255,.55);
        border: 1px dashed #c7d2e6;
        border-radius: 20px;
        margin-top: 1rem;
    }
    .hint .hint-icon {
        font-size: 2.4rem;
        margin-bottom: .4rem;
    }
    .hint h3 {
        margin: 0 0 .35rem 0;
        color: #1c2740;
    }
    .hint p {
        margin: 0;
        font-size: .92rem;
    }
    .hint-tips {
        margin-top: 1.2rem;
        display: flex;
        gap: .6rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    .hint-chip {
        background: white;
        border: 1px solid #dde5f5;
        color: #35406b;
        padding: .4rem .85rem;
        border-radius: 999px;
        font-size: .8rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(30,45,75,.05);
    }

    /* ---------- Status box tweaks ---------- */
    [data-testid="stStatusWidget"], .stStatus {
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🤖 Agentic Chatbot</h1>
        <p>Asisten AI dengan RAG, web search, dan informasi cuaca — siap membantu 24/7.</p>
        <div class="hero-badges">
            <span class="hero-badge">📄 RAG dari PDF</span>
            <span class="hero-badge">🌐 Web Search</span>
            <span class="hero-badge">⛅ Cuaca Real-time</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# ========================= Sidebar threading feature =========================

st.sidebar.markdown("## 💬 Percakapan")
st.sidebar.caption("Simpan konteks chat dalam beberapa thread.")


# Create a button for starting a new conversation
if st.sidebar.button("＋  Chat Baru", use_container_width=True):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()

st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

if st.session_state["chat_threads"]:
    st.sidebar.caption(f"📂 {len(st.session_state['chat_threads'])} thread tersimpan")
else:
    st.sidebar.caption("Belum ada riwayat percakapan.")

# Display all conversation threads in reverse order
# This shows the newest conversation first
for index, thread_id in enumerate(st.session_state["chat_threads"][::-1], start=1):

    # Create one sidebar button for every conversation
    is_active = thread_id == st.session_state["thread_id"]
    prefix = "🟢 " if is_active else "💬 "
    label = f"{prefix}Chat {index}  ·  {str(thread_id)[:8]}"
    if st.sidebar.button(label, key=thread_id, use_container_width=True):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append(
                {
                    "role": role,
                    "content": message_text(message.content),
                    "tools": [
                        call.get("name", "tool")
                        for call in getattr(message, "tool_calls", [])
                        if call.get("name")
                    ],
                }
            )

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()

st.sidebar.markdown(
    f'<div class="sidebar-footer">Agentic Chatbot • {datetime.date.today().year}</div>',
    unsafe_allow_html=True,
)


# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div class="hint">
            <div class="hint-icon">✨</div>
            <h3>Mulai percakapan baru</h3>
            <p>Tanyakan apa saja, atau unggah PDF untuk mendapatkan jawaban berbasis dokumen.</p>
            <div class="hint-tips">
                <span class="hint-chip">📄 "Ringkas isi PDF ini"</span>
                <span class="hint-chip">🌤️ "Cuaca Jakarta hari ini?"</span>
                <span class="hint-chip">🔎 "Cari berita terbaru soal AI"</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):

        # Display the message content
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_tool_history(message.get("tools", []))


# ========================= Fixed chat input with PDF upload =========================

# Keep st.chat_input directly in the main body.
# This keeps it fixed at the bottom of the screen.
#
# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input(
    "Tulis pesan, atau lampirkan PDF untuk ditanyakan...",
    accept_file=True,
    file_type=["pdf"],
)


# Default user input value
user_input = None


# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        try:

            # Save the uploaded PDF as a temporary local file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(uploaded_pdf.getvalue())

                temporary_file_path = temporary_file.name

            # Call the existing backend RAG ingestion function
            with st.spinner(f"📄 Memproses {uploaded_pdf.name}..."):

                ingest_rag_document(temporary_file_path)

            # Display PDF processing confirmation
            st.toast(f"{uploaded_pdf.name} berhasil diproses.", icon="✅")

        except Exception as error:

            # Display PDF processing error
            st.error(f"Gagal memproses PDF: {error}")

        finally:

            # Delete the temporary PDF after indexing
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)


# Run this block after the user submits a message
if user_input:
    # Display the user's message in the chat interface
    with st.chat_message("user", avatar="🧑‍💻"):
        st.text(user_input)

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    # Assistant streaming block
    with st.chat_message("assistant", avatar="🤖"):
        status_box = st.status("🧠 Menyiapkan jawaban...", expanded=True)
        tool_names = []
        seen_tool_calls = set()

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Tool calls arrive in AI chunks before the tool result, so show
                # them immediately instead of waiting for ToolMessage.
                if isinstance(message_chunk, AIMessage):
                    for call in getattr(message_chunk, "tool_calls", []):
                        name = call.get("name")
                        call_id = call.get("id") or name
                        if name and call_id not in seen_tool_calls:
                            seen_tool_calls.add(call_id)
                            tool_names.append(name)
                            status_box.write(f"🔧 Memanggil `{name}`...")
                            status_box.update(
                                label=f"🔧 Memanggil {len(tool_names)} tool",
                                state="running",
                                expanded=True,
                            )

                if isinstance(message_chunk, ToolMessage):
                    name = tool_name(message_chunk)
                    if name not in tool_names:
                        tool_names.append(name)
                    status_box.write(f"✅ `{name}` selesai")

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    text = message_text(message_chunk.content)
                    if text:
                        yield text

        ai_message = st.write_stream(ai_only_stream())

        status_box.update(
            label=(
                f"✅ Selesai • {len(tool_names)} tool digunakan"
                if tool_names
                else "✅ Jawaban selesai"
            ),
            state="complete",
            expanded=False,
        )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
            "tools": tool_names,
        }
    )
