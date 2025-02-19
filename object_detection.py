# my_yolo/yolo_model.py

import torch
import cv2
import numpy as np
import ssl

# SSL
ssl._create_default_https_context = ssl._create_unverified_context

# YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def process_image(image0):
    image = cv2.imread(image0)
    # (BGR to RGB)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run model
    results = model(img)

    # Get result as DataFrame
    df = results.pandas().xyxy[0]

    # Loop for detected objects
    for index, row in df.iterrows():
        # Coordinates
        x_min = int(row['xmin'])
        y_min = int(row['ymin'])
        x_max = int(row['xmax'])
        y_max = int(row['ymax'])

        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Create mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])  # Bottom green border
    upper_green = np.array([80, 255, 255])  # Top green border
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Filter for large contours
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
    return (x, y), h, w
