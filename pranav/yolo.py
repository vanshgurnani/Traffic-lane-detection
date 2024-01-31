import cv2
import numpy as np
import os
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open video file
video_path = "test001.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the specific region (box) coordinates
region_coordinates = [(300, 100), (800, 100), (800, 500), (300, 500)]

# Create output directory to save frames
output_directory = "output_frames"
os.makedirs(output_directory, exist_ok=True)

# Set the frame skipping interval (adjust as needed)
frame_skip_interval = 2  # Skip every 2 frames

# Set parameters for minimum green signal duration, minimum and maximum density for green signal
min_green_duration = 30  # 30 seconds in seconds
min_density_for_green = 0.65  # 65%
max_density_for_red = 0.3  # 30%

# Confidence threshold for object detection
confidence_threshold = 0.7

# Initialize variables for timing and signal status
green_start_time = None
signal_status = "RED"  # Initial status

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Preprocess the frame and perform object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Get information about detected objects
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
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

    # Draw bounding boxes on detected vehicles
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate the density of traffic in the specific region (box)
    region_mask = np.zeros_like(frame[:, :, 0])
    region_pts = np.array(region_coordinates, np.int32)
    cv2.fillPoly(region_mask, [region_pts], 255)
    region_frame = cv2.bitwise_and(frame, frame, mask=region_mask)

    vehicle_mask = np.zeros_like(frame[:, :, 0])
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(vehicle_mask, (x, y), (x + w, y + h), 255, -1)

    region_vehicle_mask = cv2.bitwise_and(vehicle_mask, region_mask)
    occupied_space = np.count_nonzero(region_vehicle_mask)
    total_space = np.count_nonzero(region_mask)
    traffic_density = occupied_space / total_space

    # Check if the minimum green signal duration has passed
    if green_start_time is not None:
        elapsed_green_duration = time.time() - green_start_time
        if elapsed_green_duration >= min_green_duration:
            green_start_time = None
            signal_status = "RED"

    # Turn signal green if traffic density is above 65%
    if signal_status == "RED" and traffic_density > min_density_for_green:
        signal_status = "GREEN"
        green_start_time = time.time()

    # Turn signal red if traffic density drops below 30%
    elif signal_status == "GREEN" and traffic_density < max_density_for_red:
        signal_status = "RED"
        green_start_time = None

    # Display the traffic signal status, traffic density, and timings on the video
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Signal: {signal_status}", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Traffic Density: {traffic_density * 100:.2f}%", (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the elapsed green time if the signal is green
    if signal_status == "GREEN":
        elapsed_green_duration = time.time() - green_start_time
        cv2.putText(frame, f"Green Time: {min_green_duration - elapsed_green_duration:.2f}s", (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        elapsed_green_duration = 0

    # Display the frame (remove this if you only want to save frames without displaying)
    cv2.imshow("Output", frame)

# Release the VideoCapture when done
cap.release()
cv2.destroyAllWindows()
