import streamlit as st
import PIL
import tempfile
import asyncio
import os
import pandas as pd

from config import Config
from logger import logger
from object_detection import ObjectDetection

from embedder import Embedder
from chatbot import Chatbot
from history import ChatHistory
from utils import disable_mainmenu, disable_sidebar

CFG = Config()

def reset_chat():
    if os.path.exists(CFG.output_file):
        os.remove(CFG.output_file)
        st.session_state["chatbot"] = None
        st.session_state["reset_chat"] = True
    if os.path.exists("./embeddings/" + CFG.output_embeddings):
        os.remove("./embeddings/" + CFG.output_embeddings)
        st.session_state["chatbot"] = None
        st.session_state["reset_chat"] = True

async def main():
    st.set_page_config(layout="wide", page_icon="", page_title="YOLO-GPT", initial_sidebar_state="collapsed")  # noqa: E501
    disable_mainmenu()
    disable_sidebar()
    st.session_state["reset_chat"] = False
    st.title("YOLO-GPT - Detect and Ask")
    col1, col2 = st.columns(2)
    with col1:
        object_detection = ObjectDetection(capture_index=0)
        object_detection.model_type = "Detection" # Detection or Segmentation
        object_detection.start()
        confidence = float(st.slider("Select YOLOv8 Model Confidence", 25, 100, 40)) / 100  # noqa: E501

        st.header("Image/Video Config")
        source_radio = st.radio("Select Source", ['Image', 'Video', 'Webcam' ])
    
        if source_radio == 'Image':
            fp = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])  # noqa: E501
            if fp is not None:
                target_image = PIL.Image.open(fp)
                placeholder = st.image(target_image, caption="Image to Use", use_column_width=True)  # noqa: E501
                if st.button("Start Object Detection", key="image"):
                    logger.info("Starting New Object Detection on Image")
                    reset_chat()
                    placeholder.empty()
                    result, num, _ = object_detection.predict(target_image, confidence)  # noqa: E501       
                    result_plot = result[0].plot()[:, :, ::-1]
                    placeholder = st.image(result_plot, caption="Result", use_column_width=True)  # noqa: E501
                    df = pd.DataFrame(num)
                    df.to_csv(CFG.output_file, index=False)
                    st.dataframe(df)
        elif source_radio == 'Video':
            fp = st.file_uploader("Upload Video", type=['mp4']) 
            if fp is not None:
                if st.button("Start Object Detection", key="video"):
                    logger.info("Starting New Object Detection on Video")
                    reset_chat()
                    tmp_file = tempfile.NamedTemporaryFile(delete=False)
                    tmp_file.write(fp.read())
                    stored_items = object_detection.play_video(tmp_file, confidence, time_limit=0)  # noqa: E501
                    df = pd.DataFrame(stored_items)
                    df.to_csv(CFG.output_file, index=False)
                    st.dataframe(df)
        elif source_radio == 'Webcam':
            if st.button("Start Object Detection", key="webcam"):
                logger.info("Starting New Object Detection on Webcam")
                reset_chat()
                stored_items = object_detection.play_video(CFG.webcam_index, confidence, time_limit=30)  # noqa: E501
                df = pd.DataFrame(stored_items)
                df.to_csv(CFG.output_file, index=False)
                st.dataframe(df)
        else:
            st.error("Please select a valid source!")
            logger.error("Please select a valid source!")
    with col2:
        temperature = float(st.slider("Select GPT Model Temperature", 0, 100, 20)) / 100  # noqa: E501
        if os.path.exists(CFG.output_file):
            uploaded_file = CFG.output_file
            history = ChatHistory()
            try:
                embedder = Embedder()
                with st.spinner("Loading Chatbot..."):
                    vectors = await embedder.getEmbeds(uploaded_file)  # noqa: E501
                    chatbot = Chatbot(CFG.model_name, temperature, vectors)
                st.session_state["ready"] = True
                st.session_state["chatbot"] = chatbot
                if st.session_state["ready"]:
                    response_container, prompt_container = st.container(), st.container()  # noqa: E501
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
