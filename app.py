import cv2
import streamlit as st
from src.detection import ObjectDetector
from src.captioning import ImageCaptioner
from src.utils import bgr_to_pil

st.set_page_config(page_title="Smart Vision Assistant", layout="wide")
st.title("📷 Smart Vision Assistant (YOLO + BLIP)")

st.sidebar.header("Controls")
run = st.sidebar.toggle("Run webcam", value=False)
caption_every = st.sidebar.slider("Caption every N frames", 5, 60, 20)

detector = ObjectDetector("yolov8n.pt")
captioner = ImageCaptioner()

frame_placeholder = st.empty()
caption_placeholder = st.empty()

cap = cv2.VideoCapture(0)

frame_count = 0
latest_caption = ""

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Could not read from webcam.")
        break

    frame_count += 1
    annotated = detector.detect(frame)

    if frame_count % caption_every == 0:
        pil_img = bgr_to_pil(frame)
        latest_caption = captioner.caption(pil_img)

    frame_placeholder.image(annotated, channels="BGR")
    caption_placeholder.markdown(f"**Caption:** {latest_caption}")

cap.release()
