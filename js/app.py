from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from base64 import b64decode, b64encode
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results
import io
import numpy as np
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

#MODEL_NAMES = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
#PRE_TRAINED_MODEL = YOLO(MODEL_NAMES[0])
PRE_TRAINED_MODEL = YOLO("/home/gerdson/projetos/FIAP-Hackathon-Grupo2/runs/detect/train/weights/best.pt")
IMG_SHAPE = [640, 480]
IMG_QUALITY = 0.8


def js_response_to_image(js_response) -> Image.Image:
    _, b64_str = js_response['img'].split(',')
    jpeg_bytes = b64decode(b64_str)
    image = Image.open(io.BytesIO(jpeg_bytes))
    return image

def turn_non_black_pixels_visible(rgba_compatible_array: np.ndarray) -> np.ndarray:
    rgba_compatible_array[:, :, 3] = (rgba_compatible_array.max(axis=2) > 0).astype(int) * 255
    return rgba_compatible_array

def black_transparent_rgba_canvas(w, h) -> np.ndarray:
    return np.zeros([w, h, 4], dtype=np.uint8)

def draw_annotations_on_transparent_bg(detection_result: Results) -> Image.Image:
    black_rgba_canvas = black_transparent_rgba_canvas(*detection_result.orig_shape)
    transparent_canvas_with_boxes_invisible = detection_result.plot(font='verdana', masks=False, img=black_rgba_canvas)
    transparent_canvas_with_boxes_visible = turn_non_black_pixels_visible(transparent_canvas_with_boxes_invisible)
    image = Image.fromarray(transparent_canvas_with_boxes_visible, 'RGBA')
    return image

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('take_photo')
def handle_take_photo(data):
    label = data.get('label', '')
    img_data = data.get('imgData', '')
    
    if not data:
       return

    if img_data != '':
         _, b64_str = img_data.split(',')
         jpeg_bytes = b64decode(b64_str)
         captured_img = Image.open(io.BytesIO(jpeg_bytes))
         for detection_result in PRE_TRAINED_MODEL(source=np.array(captured_img), verbose=False):
             annotations_img = draw_annotations_on_transparent_bg(detection_result)
             with io.BytesIO() as buffer:
                annotations_img.save(buffer, format='png')
                img_as_base64_str = str(b64encode(buffer.getvalue()), 'utf-8')
                img_data = f'data:image/png;base64,{img_as_base64_str}'
             emit('update_image', {'label': "Processing...", 'imgData': img_data})
    
    emit('capture_complete', {'label': label, 'imgData': img_data})


if __name__ == '__main__':
    socketio.run(app, debug=True)