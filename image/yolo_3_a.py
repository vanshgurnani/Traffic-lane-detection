import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open video capture
cap = cv2.VideoCapture('test3.mp4')  # Replace with your video path

# Decrease the size of the output video
output_width = 640  # Set desired width
output_height = 480  # Set desired height

density_threshold = 70

# Define trapezium coordinates as a percentage of the frame dimensions
trapezium_top_width_percentage = 40  # Adjust as needed
trapezium_height_percentage = 80  # Adjust as needed

# Function to handle mouse events (optional for interactive adjustments)
def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Coordinates: ({x}, {y})")

# Set the mouse event callback function (optional)
cv2.namedWindow('Processed Frame')
cv2.setMouseCallback('Processed Frame', mouse_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Calculate trapezium coordinates dynamically based on the frame dimensions
    trapezium_top_width = (trapezium_top_width_percentage / 100) * width
    trapezium_height = (trapezium_height_percentage / 100) * height

    area_coordinates_pixel = [
        (int((width - trapezium_top_width) / 2), 0),
        (int((width + trapezium_top_width) / 2), 0),
        (width, int(trapezium_height)),
        (0, int(trapezium_height))
    ]

    # Draw the trapezoidal red boundary on the frame
    cv2.polylines(frame, [np.array(area_coordinates_pixel, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw the grid
    grid_color = (255, 0, 0)  # Blue color in OpenCV format (B, G, R)

    # Draw horizontal lines and display grid coordinates
    for i in range(1, 10):
        y = int((i / 10) * height)
        cv2.line(frame, (0, y), (width, y), grid_color, 1)
        cv2.putText(frame, f"{i}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw vertical lines and display grid coordinates
    for i in range(1, 10):
        x = int((i / 10) * width)
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)
        cv2.putText(frame, f"{i}", (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Preprocess frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Get information about detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Define the red area
    red_area = np.array([area_coordinates_pixel], dtype=np.int32)
    red_area = red_area.reshape((-1, 1, 2))

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

                # Check if bounding box intersects with the red area
                if cv2.pointPolygonTest(red_area, (center_x, center_y), False) > 0:
                    # Object is inside the red area
                    objects_inside_red_area.append(len(boxes) - 1)

    # Implementing Non-Maximum Suppression (NMS)
    if len(boxes) > 0:
        confidences = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Show green boxes only for objects detected inside the red area
        if len(indices) > 0:
            indices = [i for i in indices.flatten() if i in objects_inside_red_area]

            num_objects_after_nms = len(indices)

            # Drawing green boxes around detected objects on the frame
            for i in indices:
                x, y, w, h = boxes[i]
                # Draw a green rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color: (0, 255, 0), Thickness: 2

            # Calculate the area of the red box using the formula for a convex quadrilateral
            x1, y1 = area_coordinates_pixel[0]
            x2, y2 = area_coordinates_pixel[1]
            x3, y3 = area_coordinates_pixel[2]
            x4, y4 = area_coordinates_pixel[3]

            red_box_area = 0.0002645833 * (0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) + x4 * (y2 - y1)))

            # Calculate density percentage
            density_percentage = (num_objects_after_nms / red_box_area) * 100
            
            if density_percentage >= density_threshold:
                light_color = (0, 255, 0)  # Green
                light_text = "Green Light"
            else:
                light_color = (0, 0, 255)  # Red
                light_text = "Red Light"

            # Draw the area and density text on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            area_text = f"Red Box Area: {red_box_area:.2f} sq. unit"
            density_text = f"Density: {density_percentage:.2f}%"
            light_text = f"Traffic Light: {light_text}"
            cv2.putText(frame, area_text, (10, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, density_text, (10, 90), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, light_text, (10, 120), font, 0.7, light_color, 2, cv2.LINE_AA)

            # Draw count text on the frame
            count_text = f"Total Vehicle Count: {num_objects_after_nms}"
            cv2.putText(frame, count_text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Road detection using color-based segmentation (example using yellow color for roads)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    road_detected = cv2.bitwise_and(frame, frame, mask=mask)

    # Resize the frame
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Display the processed frame without saving
    cv2.imshow('Processed Frame', resized_frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()