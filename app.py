import streamlit as st

if "counter" not in st.session_state:
    st.session_state["counter"] = 0

if st.button("Tambah"):
    st.session_state["counter"] += 1

st.write(st.session_state["counter"])
