# Agentic Chatbot

Project ini adalah chatbot sederhana berbasis LangGraph dan Streamlit.
User mengirim pesan, lalu pesan diproses oleh node `chatbot` untuk menghasilkan jawaban dari model LLM.

## List Fitur

- Alur agent sederhana berbasis LangGraph (`START -> chatbot -> END`).
- Antarmuka chat interaktif menggunakan Streamlit.
- Dukungan memory persistence dengan SQLite (riwayat percakapan tersimpan).
- Streaming response dari model agar jawaban tampil bertahap (real-time).
- Struktur yang mudah dikembangkan untuk penambahan tool, multi-node, dan workflow yang lebih kompleks.

## Dokumentasi Streamlit (`st.session_state` / `ss`)

Pada aplikasi Streamlit, state percakapan disimpan di `st.session_state` (sering disingkat `ss`) agar data tetap tersedia antar rerun.

Tampilan UI Streamlit:

![UI Streamlit](images/ui-streamlit.png)

Contoh penggunaan umum:

```python
import streamlit as st

ss = st.session_state
if "messages" not in ss:
    ss.messages = []

ss.messages.append({"role": "user", "content": "Halo"})
```

Dengan pola ini:
- Riwayat chat tidak hilang saat tombol diklik atau input baru dikirim.
- Data UI (mis. pesan, status loading, thread id) tetap konsisten selama sesi user.

## Workflow Graph

Alur di bawah menunjukkan proses inti yang saat ini masih sederhana: dari `START` ke `chatbot`, lalu ke `END`.

```mermaid
flowchart TD
    A([START]) --> B[chatbot]
    B --> C([END])

```
