import logging
import os

import cv2
import gradio as gr
from gradio_webrtc import WebRTC
from ultralytics import YOLO

# Configure the root logger to WARNING to suppress debug messages from other libraries
logging.basicConfig(level=logging.WARNING)

# Create a console handler
#console_handler = logging.FileHandler("gradio_webrtc.log")
#console_handler.setLevel(logging.DEBUG)

# Create a formatter
#formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#console_handler.setFormatter(formatter)

# Configure the logger for your specific library
#logger = logging.getLogger("gradio_webrtc")
#logger.setLevel(logging.DEBUG)
#logger.addHandler(console_handler)

model = YOLO("./runs/detect/train/weights/best.pt") 

account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

def detection(image, conf_threshold=0.3):
    print("Detectando objetos...")
    image = cv2.resize(image, (model.input_width, model.input_height))
    new_image = model.detect_objects(image, conf_threshold)
    return cv2.resize(new_image, (500, 500))


css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10 Webcam Stream (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(
                label="Stream",
                mode="send-receive",
                modality="video",
                track_constraints={
                    "width": {"exact": 800},
                    "height": {"exact": 600},
                    "aspectRatio": {"exact": 1.33333},
                },
                rtp_params={"degradationPreference": "maintain-resolution"},
            )
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )
            number = gr.Number()

        image.stream(
            fn=detection, inputs=[image, conf_threshold], outputs=[image] 
        )

demo.launch()