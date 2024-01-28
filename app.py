from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

net = cv2.dnn.readNet('yolov4.cfg', 'yolov4.weights')
classes = []
with open('yolov4.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    img_stream = file.read()
    nparr = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    people_counter = 0

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.75 and class_id == 0: 
                people_counter += 1
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_encoded = base64.b64encode(buffer).decode('utf-8')

    return render_template('result.html', img_data=img_encoded, people_counter=people_counter)

if __name__ == '__main__':
    app.run(debug=True)
