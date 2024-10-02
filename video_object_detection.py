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

    # Convert the OpenCV frame to the format required by the YOLO model (BGR to RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the model and get results
    results = model(img)

    # Get the results as a DataFrame
    df = results.pandas().xyxy[0]
    frame_width = frame.shape[1]
    frame_mid_x = frame_width // 2
    is_mid = False

    # If at least one object is detected
    if not df.empty:
        # Calculate the area to find the largest rectangle
        df['area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])
        
        # Select the largest rectangle
        largest_obj = df.loc[df['area'].idxmax()]

        # Coordinates
        x_min = int(largest_obj['xmin'])
        y_min = int(largest_obj['ymin'])
        x_max = int(largest_obj['xmax'])
        y_max = int(largest_obj['ymax'])
        confidence = largest_obj['confidence']
        width = x_max - x_min
        height = y_max - y_min

        # Check confidence score
        if confidence > 0.4:
            # Draw the rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            rect_mid_x = (x_min + x_max) // 2
            rect_mid_y = (y_min + y_max) // 2

            # Draw a red circle at the midpoint of the rectangle
            cv2.circle(frame, (rect_mid_x, rect_mid_y), 5, (0, 0, 255), -1)

            # If the midpoint of the rectangle is near the center of the frame, set "is_mid" to True
            if abs(rect_mid_x - frame_mid_x) < 100:  # 100 pixels proximity tolerance
                is_mid = True

            # Write the coordinates at the top left corner of the rectangle
            label = f"({x_min},{y_min})"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Add text for the other corners
            cv2.putText(frame, f"({x_max},{y_min})", (x_max - 80, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"({x_min},{y_max})", (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"({x_max},{y_max})", (x_max - 80, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # 
            cv2.putText(frame, f"W: {width}", (x_min + width // 2 - 30, y_max - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, f"H: {height}", (x_min - 80, y_min + height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    if is_mid:
        cv2.putText(frame, "True", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display the results
    cv2.imshow('YOLOv5 Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
