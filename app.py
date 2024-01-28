from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import base64

app = Flask(__name__)

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

    # Odczytaj obraz z pliku
    img_stream = file.read()
    nparr = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detekcja osób przy użyciu OpenCV
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(16, 16), scale=1.05)

    # Narysuj prostokąty wokół wykrytych osób
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Zapisz obraz z prostokątami
    _, buffer = cv2.imencode('.jpg', img)
    img_encoded = base64.b64encode(buffer).decode('utf-8')

    return render_template('result.html', img_data=img_encoded)

if __name__ == '__main__':
    app.run(debug=True)
