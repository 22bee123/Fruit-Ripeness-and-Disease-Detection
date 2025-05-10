from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import time
import os
import signal
import sys

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Global camera variable
camera = None

def get_camera():
    """Initialize and return camera object"""
    global camera
    
    # If camera is already initialized, return it
    if camera is not None and camera.isOpened():
        return camera
    
    # Try to initialize camera with different backends on Windows
    camera_backends = [
        cv2.CAP_DSHOW,  # DirectShow (should be the most reliable on Windows)
        cv2.CAP_MSMF,   # Microsoft Media Foundation
        cv2.CAP_ANY     # Any available API
    ]
    
    # Try different camera indices with different backends
    for backend in camera_backends:
        for idx in range(2):  # Try camera index 0 and 1
            print(f"Trying to open camera {idx} with backend {backend}")
            try:
                # Use specific backend for Windows
                cam = cv2.VideoCapture(idx, backend)
                
                # Set camera properties
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cam.set(cv2.CAP_PROP_FPS, 30)
                
                # Check if camera is opened
                if cam.isOpened():
                    print(f"Successfully opened camera {idx} with backend {backend}")
                    print(f"Resolution: {cam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cam.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                    camera = cam
                    return camera
                else:
                    cam.release()
            except Exception as e:
                print(f"Error opening camera {idx} with backend {backend}: {str(e)}")
    
    print("Failed to open any camera")
    return None

# Clean up function to release camera on exit
def cleanup_camera():
    global camera
    if camera is not None:
        camera.release()
        print("Camera released on exit")

# Register cleanup function
def signal_handler(sig, frame):
    cleanup_camera()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Load YOLO model for fruit detection
fruit_detection_model = YOLO("weights_3/best.pt")  # Change this path to the path of your YOLO model
banana_disease_detection_model = YOLO(
    "train2/weights/best.pt")  # Path to YOLOv8 model for banana disease detection
mango_disease_detection_model = YOLO(
    "train/weights/best.pt")  # Path to YOLOv8 model for mango disease detection
pomogranate_disease_detection_model = YOLO(
    "train4/weights/best.pt")  # Path to YOLOv8 model for pomogranate disease detection


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/fruit_detection')
def fruit_detection():
    return render_template('fruit_detection.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Receive image data from the client
    image_data = request.json['image_data'].split(',')[1]  # Remove the data URL prefix

    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)

    # Convert image bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Perform object detection using YOLO
    results = fruit_detection_model(image)

    # Extract detection results
    detected_objects = []
    for result in results:
        boxes = result.boxes.xywh.cpu()  # xywh bbox list
        clss = result.boxes.cls.cpu().tolist()  # classes Id list
        names = result.names  # classes names list
        confs = result.boxes.conf.float().cpu().tolist()  # probabilities of classes

        for box, cls, conf in zip(boxes, clss, confs):
            detected_objects.append({'class': names[cls], 'bbox': box.tolist(), 'confidence': conf})

    return jsonify(detected_objects)


@app.route('/disease_detection')
def disease_detection():
    return render_template('disease_detection.html')


@app.route('/banana_detection', methods=['GET', 'POST'])
def banana_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(banana_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('banana_detection.html')


@app.route('/mango_detection', methods=['GET', 'POST'])
def mango_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(mango_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('mango_detection.html')

@app.route('/pomogranate_detection', methods=['GET', 'POST'])
def pomogranate_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(pomogranate_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('pomogranate_detection.html')


# New routes for real-time disease detection
@app.route('/realtime_banana_disease')
def realtime_banana_disease():
    return render_template('realtime_disease.html', fruit_type='banana')

@app.route('/realtime_mango_disease')
def realtime_mango_disease():
    return render_template('realtime_disease.html', fruit_type='mango')

@app.route('/realtime_pomogranate_disease')
def realtime_pomogranate_disease():
    return render_template('realtime_disease.html', fruit_type='pomogranate')

@app.route('/video_feed_disease/<fruit_type>')
def video_feed_disease(fruit_type):
    return Response(generate_disease_frames(fruit_type), mimetype='multipart/x-mixed-replace; boundary=frame')


def detect_disease(model, image):
    result = model(image)
    class_names = []
    for result in result:
        probs = result.probs
        class_index = probs.top1
        class_name = result.names[class_index]
        score = float(probs.top1conf.cpu().numpy())
        class_names.append(class_name)
    return class_names


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return render_template('uploaded_image.html', filename=filename)


def generate_disease_frames(fruit_type):
    # Select the appropriate model based on fruit type
    if fruit_type == 'banana':
        model = banana_disease_detection_model
    elif fruit_type == 'mango':
        model = mango_disease_detection_model
    elif fruit_type == 'pomogranate':
        model = pomogranate_disease_detection_model
    else:
        # Default to banana if unknown fruit type
        model = banana_disease_detection_model
    
    # Get camera
    cam = get_camera()
    
    if cam is None:
        print("Error: Could not open any camera. Please check your camera connection.")
        # Generate a placeholder frame with error message
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera not available", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()
        
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)  # Yield the error frame every second
        return
    
    try:
        while True:
            success, frame = cam.read()
            
            if not success:
                print("Failed to capture frame")
                # If frame capture fails, wait a bit and try again
                time.sleep(0.1)
                continue
            
            # Important: OpenCV uses BGR color format, but we need RGB for processing
            # No conversion here - keep in BGR for OpenCV operations
            
            try:
                # Convert to RGB for the model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image for classification
                pil_image = Image.fromarray(frame_rgb)
                
                # Process frame with disease detection model
                # Use task='classify' to ensure the model performs classification instead of detection
                results = model(pil_image, task='classify')
                
                # Get the prediction
                disease_name = "Unknown"
                confidence = 0
                
                for result in results:
                    # Check if probs attribute exists
                    if hasattr(result, 'probs') and result.probs is not None:
                        class_index = result.probs.top1
                        disease_name = result.names[class_index]
                        confidence = float(result.probs.top1conf.cpu().numpy())
                        print(f"Detected: {disease_name} with confidence {confidence}")
                    else:
                        print("No probs attribute found in result")
                
                # Draw the results on the frame - work with BGR (OpenCV default)
                # Add a colored background for the text
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Add text with disease name and confidence - using BGR color format
                # Green in BGR is (0, 255, 0)
                cv2.putText(frame, f"Fruit: {fruit_type.capitalize()}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Disease: {disease_name}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add timestamp - white in BGR is (255, 255, 255)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (frame.shape[1] - 250, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # IMPORTANT: For JPEG encoding, we need to keep the BGR format
                # DO NOT convert to RGB before encoding to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    print("Failed to encode frame")
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                import traceback
                traceback.print_exc()
                # Return the original frame on error
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except GeneratorExit:
        print("Generator exited")
    except Exception as e:
        print(f"Unexpected error in generate_disease_frames: {str(e)}")
        traceback.print_exc()


def generate_frames():
    # Get camera
    cam = get_camera()
    
    if cam is None:
        print("Error: Could not open any camera. Please check your camera connection.")
        # Generate a placeholder frame with error message
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera not available", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()
        
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)  # Yield the error frame every second
        return
    
    try:
        while True:
            success, frame = cam.read()
            
            if not success:
                print("Failed to capture frame")
                # If frame capture fails, wait a bit and try again
                time.sleep(0.1)
                continue
            
            # Keep in BGR format for OpenCV operations
            try:
                # Convert to RGB for YOLO model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with YOLO model using RGB frame
                fruit_results = fruit_detection_model(frame_rgb, conf=0.4)
                
                # Draw detection results on the frame
                for result in fruit_results:
                    # result.plot() returns RGB image with detections
                    annotated_frame = result.plot()
                    
                    # Convert back to BGR for OpenCV operations and correct color display
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add timestamp - using BGR color format
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(annotated_frame_bgr, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # IMPORTANT: For JPEG encoding, we need to keep the BGR format
                    # DO NOT convert to RGB before encoding to JPEG
                    ret, buffer = cv2.imencode('.jpg', annotated_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        print("Failed to encode frame")
                        continue
                        
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                # Return the original frame on error
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except GeneratorExit:
        print("Generator exited")
    except Exception as e:
        print(f"Unexpected error in generate_frames: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    # Make sure camera is released when app exits
    try:
        # Run Flask app
        app.run(host="0.0.0.0", debug=False, threaded=True)
    finally:
        cleanup_camera()
