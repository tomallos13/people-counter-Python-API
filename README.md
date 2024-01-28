# Object Detection Web App

This is a simple web application for object detection using YOLO (You Only Look Once) model. It allows users to upload an image file or provide a URL for object detection.

![Object Detection]('detector.jpeg')

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/tomallos/people-counter-Python-API.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download model:
    ```bash
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg 
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights 
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.names
     ```

4. Run the application:

    ```bash
    python app.py
    ```

5. Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)



## Usage

- Upload an image file by choosing "Detect from File."
- Provide a URL and click "Detect from URL."

## Requirements

- Flask==2.0.1
- opencv-python==4.5.3.56
- requests==2.26.0

## Acknowledgments

- YOLO model: [YOLOv4](https://github.com/AlexeyAB/darknet)