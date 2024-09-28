import time
import torch
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Global variable to track human entry
human_entered = False

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def detect_human_in_zone(img, zone_x, zone_y, zone_w, zone_h):
    global human_entered
    
    # Perform inference
    results = model(img)
    
    # Parse results
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        label = model.names[int(cls)]
        
        # Check if the detected object is a person and within the zone
        if label == 'person':
            if x1 > zone_x and y1 > zone_y and x2 < (zone_x + zone_w) and y2 < (zone_y + zone_h):
                human_entered = True
                print("Human entered the zone!")
                return  # No need to continue checking once a human is detected
    
    human_entered = False

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    
    # Define the square zone (top-left corner and size)
    zone_x, zone_y = 100, 100
    zone_w, zone_h = 500, 500

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            
            # Detect if a human enters the zone
            detect_human_in_zone(img, zone_x, zone_y, zone_w, zone_h)
            
            if human_entered:
                # Tint the frame red if a human is detected in the zone
                red_tint = img.copy()
                red_tint[:, :, 0] = 0  # Zero out the blue channel
                red_tint[:, :, 1] = 0  # Zero out the green channel
                img = red_tint
            
            # Encode the frame
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else: 
            break

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
