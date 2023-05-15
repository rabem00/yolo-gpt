import streamlit as st
import asyncio

# local imports
from config import Config
from logger import logger
from embedder import Embedder
from chatbot import Chatbot
from history import ChatHistory
from utils import handle_upload

CFG = Config()

async def main():
    st.title("YOLO-GPT - Detect and Ask")
    st.sidebar.subheader("GPT Configuration")
    temperature = float(st.sidebar.slider("Select GPT Model Temperature", 0, 100, 20)) / 100  # noqa: E501
    uploaded_file = handle_upload()
    if uploaded_file:
        history = ChatHistory()
        try:
            embedder = Embedder()
            with st.spinner("Loading Chatbot..."):
                uploaded_file.seek(0)
                file = uploaded_file.read()
                vectors = await embedder.getEmbeds(file, uploaded_file.name)
                chatbot = Chatbot(CFG.model_name, temperature, vectors)
            st.session_state["ready"] = True
            st.session_state["chatbot"] = chatbot
            if st.session_state["ready"]:
                response_container, prompt_container = st.container(), st.container()
                with prompt_container:
                    is_ready, user_input = Chatbot.prompt_form()
                    history.initialize(uploaded_file)
                    if st.session_state["reset_chat"]:
                        history.reset(uploaded_file)
                    if is_ready:
                        history.append("user", user_input)
                        output = await st.session_state["chatbot"].conversational_chat(user_input)  # noqa: E501
                        history.append("assistant", output)
                history.generate_messages(response_container)
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())