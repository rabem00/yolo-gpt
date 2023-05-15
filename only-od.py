import streamlit as st
import PIL
import tempfile
import asyncio

from config import Config
from logger import logger
from object_detection import ObjectDetection

CFG = Config()

async def main():
    
    st.title("YOLO-GPT - Detect and Ask")
 
    # Object Detection Configuration
    object_detection = ObjectDetection(capture_index=0)
    st.sidebar.subheader("YOLOv8 Configuration")
    object_detection.model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])  # noqa: E501
    object_detection.start()
    
    confidence = float(st.sidebar.slider("Select YOLOv8 Model Confidence", 25, 100, 40)) / 100  # noqa: E501
    source_img = None
    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio("Select Source", ['Image', 'Video', 'Webcam'])
 
    if source_radio == 'Image':
        fp = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])  # noqa: E501
        if fp is not None:
            target_image = PIL.Image.open(fp)
            placeholder = st.image(target_image, caption="Image to Use", use_column_width=True)  # noqa: E501
            if st.sidebar.button("Start Object Detection"):
                logger.info("Starting New Object Detection on Image")
                placeholder.empty()
                result = object_detection.predict(target_image, confidence)  # noqa: E501       
                result_plot = result[0].plot()[:, :, ::-1]
                placeholder = st.image(result_plot, caption="Result", use_column_width=True)  # noqa: E501
    elif source_radio == 'Video':
        fp = st.sidebar.file_uploader("Upload Video", type=['mp4']) 
        if fp is not None:
            if st.sidebar.button("Start Object Detection"):
                logger.info("Starting New Object Detection on Video")
                tmp_file = tempfile.NamedTemporaryFile(delete=False)
                tmp_file.write(fp.read())
                object_detection.play_video(tmp_file)
    elif source_radio == 'Webcam':
        source_img = CFG.webcam_index
        if st.sidebar.button("Start Object Detection"):
            logger.info("Starting New Object Detection on Webcam")
            object_detection.play_video(source_img)
    else:
        st.error("Please select a valid source!")
        logger.error("Please select a valid source!")

if __name__ == '__main__':
    asyncio.run(main())
