import cv2
import numpy as np
import csv
import random

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Read image
img = cv2.imread("image.jpeg")
height, width, _ = img.shape

# Display grids in blue color
grid_color = (255, 0, 0)  # Blue color in OpenCV format (B, G, R)

# Draw horizontal lines
num_horizontal_lines = 10
for i in range(1, num_horizontal_lines):
    y = int((i / num_horizontal_lines) * height)
    cv2.line(img, (0, y), (width, y), grid_color, 1)

# Draw vertical lines
num_vertical_lines = 10
for i in range(1, num_vertical_lines):
    x = int((i / num_vertical_lines) * width)
    cv2.line(img, (x, 0), (x, height), grid_color, 1)

# Generate random area with yellow boundary
area_x = 100  # X-coordinate of the area
area_y = 100  # Y-coordinate of the area
area_width = 200  # Width of the area
area_height = 150  # Height of the area

# Generate random points along the boundary
boundary_points = []
for i in range(area_x, area_x + area_width):
    boundary_points.append((i, area_y))
    boundary_points.append((i, area_y + area_height - 1))

for i in range(area_y, area_y + area_height):
    boundary_points.append((area_x, i))
    boundary_points.append((area_x + area_width - 1, i))

# Shuffle the points to get a random boundary
random.shuffle(boundary_points)

# Draw the yellow boundary on the image
for point in boundary_points:
    img[point[1], point[0]] = (0, 0, 255)  # Red color in OpenCV format (B, G, R)

# Preprocess image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(layer_names)

# Get information about detected objects
class_ids = []
confidences = []
boxes = []

# Define the red area
red_area_x = 100
red_area_y = 100
red_area_width = 200
red_area_height = 150

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

        # Drawing green boxes around detected objects on the image
        for i in indices:
            x, y, w, h = boxes[i]
            # Draw a green rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color: (0, 255, 0), Thickness: 2

        # Count the number of detected objects after NMS
        num_objects_after_nms = len(indices)

        # Draw count text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        count_text = f"Total Vehicles: {num_objects_after_nms}"
        cv2.putText(img, count_text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Save the image with grids, yellow boundary, green boxes, and red area
        cv2.imwrite("output_image_with_grids_yellow_boundary_green_boxes_and_red_area.jpg", img)
        print("Image with grids, random yellow boundary, green boxes (inside red area), red area, and count text saved as 'output_image_with_grids_yellow_boundary_green_boxes_and_red_area.jpg'")
        
    else:
        print("No valid indices found inside the red area after applying NMS.")
else:
    print("No boxes detected to apply NMS.")
