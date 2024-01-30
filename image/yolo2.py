import cv2
import numpy as np
import csv
import os
import random

# Load YOLO
net = cv2.dnn.readNet("image/yolov3.weights", "image/yolov3.cfg")
classes = []

with open("image/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open video capture
cap = cv2.VideoCapture('image/testvideo3.mp4')  # Replace 'path_to_your_video.mp4' with the actual path to your video file

# Decrease the size of the output video
output_width = 640  # Set the desired width for the output video
output_height = 480  # Set the desired height for the output video

# Define the area using coordinates
area_coordinates = [(300, 200), (800, 700)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Display grids in blue color
    grid_color = (255, 0, 0)  # Blue color in OpenCV format (B, G, R)

    # Draw horizontal lines
    num_horizontal_lines = 10
    for i in range(1, num_horizontal_lines):
        y = int((i / num_horizontal_lines) * height)
        cv2.line(frame, (0, y), (width, y), grid_color, 1)

    # Draw vertical lines
    num_vertical_lines = 10
    for i in range(1, num_vertical_lines):
        x = int((i / num_vertical_lines) * width)
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)

    # Generate points for the area using the specified coordinates
    area_x, area_y = area_coordinates[0]
    area_width = area_coordinates[1][0] - area_coordinates[0][0]
    area_height = area_coordinates[1][1] - area_coordinates[0][1]

    # Generate points along the boundary within the frame dimensions
    boundary_points = []
    for i in range(area_x, min(area_x + area_width, width)):
        boundary_points.append((i, max(area_y, 0)))
        boundary_points.append((i, min(area_y + area_height - 1, height - 1)))

    for i in range(area_y, min(area_y + area_height, height)):
        boundary_points.append((max(area_x, 0), i))
        boundary_points.append((min(area_x + area_width - 1, width - 1), i))

    # Shuffle the points to get a random boundary
    random.shuffle(boundary_points)

    # Draw the yellow boundary on the frame
    for point in boundary_points:
        frame[point[1], point[0]] = (0, 0, 255)  # Red color in OpenCV format (B, G, R)

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Get information about detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Define the red area
    red_area_x = area_x
    red_area_y = area_y
    red_area_width = area_width
    red_area_height = area_height

    # Initialize a list to store objects inside the red area
    objects_inside_red_area = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # Check if the bounding box intersects with the red area
                if (
                    x >= red_area_x
                    and y >= red_area_y
                    and x + w <= red_area_x + red_area_width
                    and y + h <= red_area_y + red_area_height
                ):
                    # Object is inside the red area
                    objects_inside_red_area.append(len(boxes) - 1)

    # Implementing Non-Maximum Suppression (NMS)
    if len(boxes) > 0:
        # Extract confidences and indices of the boxes
        confidences = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Show green boxes only for objects detected inside the red area
        if len(indices) > 0:
            indices = [i for i in indices.flatten() if i in objects_inside_red_area]

            # Count the number of detected objects after NMS
            num_objects_after_nms = len(indices)

            # Drawing green boxes around detected objects on the frame
            for i in indices:
                x, y, w, h = boxes[i]
                # Draw a green rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color: (0, 255, 0), Thickness: 2

            # Draw count text on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            count_text = f"Total Vehicles: {num_objects_after_nms}"
            cv2.putText(frame, count_text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Resize the frame
            resized_frame = cv2.resize(frame, (output_width, output_height))

            # Display the processed frame without saving
            cv2.imshow('Processed Frame', resized_frame)

        else:
            print("No valid indices found inside the red area after applying NMS.")
    else:
        print("No boxes detected to apply NMS.")

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
