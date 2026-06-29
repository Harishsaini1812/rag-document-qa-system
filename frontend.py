import streamlit as st
import requests

# ===================== CONFIG =====================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Document Q&A Assistant")


# ===================== SESSION STATE =====================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ===================== SIDEBAR =====================
st.sidebar.header("📄 Document Upload")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Process PDF"):
        files = {"file": uploaded_file.getvalue()}

        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            data={"thread_id": st.session_state.thread_id}
        )

        if response.status_code == 200:
            st.sidebar.success("PDF uploaded & processed!")
        else:
            st.sidebar.error("Upload failed")


# ===================== CHAT UI =====================

st.subheader("💬 Chat: First Upload the Document")

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_input = st.chat_input("Ask something from your document...")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # call backend
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "thread_id": st.session_state.thread_id,
            "question": user_input
        },
        stream=True
    )

    answer = ""

    with st.chat_message("assistant"):

        placeholder = st.empty()

        for chunk in response.iter_content(chunk_size=None):

            if chunk:

                text = chunk.decode("utf-8")

                answer += text

                placeholder.markdown(answer)

    # show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
