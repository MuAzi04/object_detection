import torch
import cv2
import numpy as np
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start the camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the OpenCV frame to the appropriate format for the YOLO model (convert BGR to RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the model and get the results
    results = model(img)

    # Get the results as a DataFrame
    df = results.pandas().xyxy[0]
    
    frame_width = frame.shape[1]
    frame_mid_x = frame_width // 2  # Middle point of the screen (x-axis)

    # Initialize the True/False condition
    is_mid = False

    # Loop through the detected objects (but stop after the first valid detection)
    for index, row in df.iterrows():
        # Coordinates
        x_min = int(row['xmin'])
        y_min = int(row['ymin'])
        x_max = int(row['xmax'])
        y_max = int(row['ymax'])
        confidence = row['confidence']

        # If the confidence score is high, draw the rectangle
        if confidence > 0.2:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Calculate the center of the rectangle
            rect_mid_x = (x_min + x_max) // 2
            rect_mid_y = (y_min + y_max) // 2

            # Draw a red dot at the center of the rectangle
            cv2.circle(frame, (rect_mid_x, rect_mid_y), 5, (0, 0, 255), -1)

            # If the center of the rectangle is close to the middle of the frame, set "True"
            if abs(rect_mid_x - frame_mid_x) < 100:  # 50 pixel proximity tolerance
                is_mid = True

            # Stop processing further objects after the first one is detected and processed
            break

    # Show the results
    cv2.imshow('YOLOv5 Detection', frame)

    # Create a mask for green color (green range in BGR format)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])  # Lower green limit
    upper_green = np.array([80, 255, 255])  # Upper green limit
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Only take large contours
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter out small noise by thresholding the area
            # Draw a rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Write coordinates on the corners of the rectangle
            cv2.putText(frame, f"({x},{y})", (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, f"({x + w},{y})", (x + w + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, f"({x},{y + h})", (x - 50, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, f"({x + w},{y + h})", (x + w + 5, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Write width and height in the center of the rectangle
            cv2.putText(frame, f"W: {w}", (x + w // 2 - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, f"H: {h}", (x - 80, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # Stop processing further contours after the first one is detected and processed
            break

    # Display the True/False message in the top-left corner
    if is_mid:
        cv2.putText(frame, "True", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show the results
    cv2.imshow('YOLOv5 Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
