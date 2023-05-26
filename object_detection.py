import torch
import numpy as np
import cv2
import os
import urllib.request
from time import time
#from ultralytics import YOLO
import ultralytics
import streamlit as st
import supervision as sv

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from config import Config

CFG = Config()

class ObjectDetection:

    def __init__(self, capture_index):
        self.model_type = ""
        self.download_models()
        #ultralytics.checks()
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor_cap = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model_cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    def start(self):
        if self.model_type == "Detection":
            self.model = self.load_model(CFG.models_path + CFG.detection_model)
        elif self.model_type == "Segmentation":
            self.model = self.load_model(CFG.models_path + CFG.segmentation_model)
        
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)  # noqa: E501
    
    def download_models(self):
        if not os.path.exists(CFG.models_path):
            os.makedirs(CFG.models_path)
        if not os.path.exists(CFG.models_path + CFG.detection_model):
            urllib.request.urlretrieve(CFG.download_models + CFG.detection_model, CFG.models_path + CFG.detection_model)  # noqa: E501
        if not os.path.exists(CFG.models_path + CFG.segmentation_model):
            urllib.request.urlretrieve(CFG.download_models + CFG.segmentation_model, CFG.models_path + CFG.segmentation_model)  # noqa: E501
     
    def load_model(self, path):
        model = ultralytics.YOLO(path)
        #model.fuse()
        return model

    def predict(self, frame, confidence):
        num_class_ids = []
        results = self.model.predict(frame, conf=confidence)
        for result in results:
            detection_count = result.boxes.shape[0]
            for box in result.boxes:
                num_class_ids.append(result.names[box.cls[0].item()])
        return results, num_class_ids, detection_count
   
    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            if class_id == 0:    
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
 
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]
        
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)  # noqa: E501
        return frame    
    
    def play_video(self, video, confidence, time_limit):
        if hasattr(video, 'name'):
            vf = cv2.VideoCapture(video.name)
        else:
            vf = cv2.VideoCapture(video)
        assert vf.isOpened(), 'The provided source cannot be captured.'

        vf.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        vf.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        stframe = st.empty()
        store = []
        counter = 0
        if time_limit == 0:
            no_limit = True
        else:
            no_limit = False

        count = 0
        while vf.isOpened():
            start_time = time()
            ret, frame = vf.read()
            if not ret or (counter >= time_limit and not no_limit):
                stframe.empty()
                return store
            if count % 10 == 0:
                results, num_class_ids, detection_count = self.predict(frame, confidence)
                to_store = set(num_class_ids)

                raw_image = Image.fromarray(frame).convert('RGB')
                text = "a picture of"
                inputs = self.processor_cap(raw_image, text, return_tensors="pt")
                out = self.model_cap.generate(**inputs)
                print(self.processor_cap.decode(out[0], skip_special_tokens=True))
                print(to_store)
                # Add to_store with self.processor_cap
                to_add = self.processor_cap.decode(out[0], skip_special_tokens=True) + " " + str(to_store)
                store.append(to_add)
            count += 1
            #st.write("Number of detections: ", detection_count)
            frame = self.plot_bboxes(results, frame)
            counter += 1
            
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # noqa: E501
            stframe.image(frame, channels="BGR")
