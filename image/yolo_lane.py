import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("image/yolov3.weights", "image/yolov3.cfg")
classes = []

with open("image/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open video capture
cap = cv2.VideoCapture('image/test1.mp4')  # Replace with your video path

# Decrease the size of the output video
output_width = 640  # Set desired width
output_height = 480  # Set desired height

density_threshold = 70

# Define pixel coordinates for the trapezoidal area (right lane)
area_coordinates_pixel = [(500, 240), (758, 240), (1200, 700), (200, 700)]  # Adjust as needed

# Function to adjust the trapezoidal area based on detected lanes
def adjust_trapezoidal_area_based_on_lanes(frame, lines):
    if lines is not None and len(lines) > 0:
        # Find the average slope of detected lanes
        slopes = [(line[0][3] - line[0][1]) / (line[0][2] - line[0][0] + 1e-10) for line in lines]
        avg_slope = np.mean(slopes)

        # Calculate new vertices for the trapezoidal area based on the average slope
        trapezoidal_height = 200  # Adjust as needed
        top_left = (width // 2 - int(trapezoidal_height / (2 * avg_slope)), 240)
        top_right = (width // 2 + int(trapezoidal_height / (2 * avg_slope)), 240)
        bottom_right = (width, height)
        bottom_left = (0, height)

        # Draw the adjusted trapezoidal red boundary on the frame
        cv2.polylines(frame, [np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)],
                      isClosed=True, color=(0, 0, 255), thickness=2)

# Set up a callback function for mouse events (optional for interactive adjustments)
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

            # Detect lanes and adjust parameters
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)

            # Adjust the trapezoidal area based on detected lanes
            adjust_trapezoidal_area_based_on_lanes(frame, lines)

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
