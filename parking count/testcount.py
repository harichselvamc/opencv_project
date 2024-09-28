import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Mouse callback function to capture BGR color
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a window for the video
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the video
cap = cv2.VideoCapture('test.mp4')

# Load class names from the COCO dataset
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define areas for detection as polygons
areas = [
    [(52,364),(30,417),(73,412),(88,369)],
    [(105,353),(86,428),(137,427),(146,358)],
    [(159,354),(150,427),(204,425),(203,353)],
    [(217,352),(219,422),(273,418),(261,347)],
    [(274,345),(286,417),(338,415),(321,345)],
    [(336,343),(357,410),(409,408),(382,340)],
    [(396,338),(426,404),(479,399),(439,334)],
    [(458,333),(494,397),(543,390),(495,330)],
    [(511,327),(557,388),(603,383),(549,324)],
    [(564,323),(615,381),(654,372),(596,315)],
    [(616,316),(666,369),(703,363),(642,312)],
    [(674,311),(730,360),(764,355),(707,308)]
]

# Main video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for processing
    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO object detection on the frame
    results = model.predict(frame)
    boxes_data = results[0].boxes.data
    px = pd.DataFrame(boxes_data).astype("float")
    
    # Initialize lists for counting cars in areas
    area_counters = [[] for _ in range(12)]
    
    for _, row in px.iterrows():
        x1, y1, x2, y2, conf, cls = map(int, row)
        class_name = class_list[cls]
        
        if 'car' in class_name:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box

            # Check which areas the center of the car falls into
            for i, area in enumerate(areas):
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    area_counters[i].append(class_name)
                    cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Display the polygons and their counts
    for i, area in enumerate(areas):
        color = (0, 0, 255) if len(area_counters[i]) == 1 else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, str(i + 1), (area[0][0], area[0][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    # Calculate available space and display it
    occupied = sum(1 for area_list in area_counters if area_list)
    space = 12 - occupied
    cv2.putText(frame, f'Space: {space}', (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('RGB', frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
