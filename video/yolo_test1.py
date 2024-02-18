import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open video capture
cap = cv2.VideoCapture('video.mp4')  # Replace with your video path

# Decrease the size of the output video
output_width = 640  # Set desired width
output_height = 480  # Set desired height

density_threshold = 70

# Frame skipping configuration
skip_frames = 5  # Adjust as needed

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % skip_frames != 0:
        continue  # Skip frames

    # Lane detection using Canny edge detection and Hough line transformation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Resize the edges frame
    resized_edges = cv2.resize(edges, (output_width, output_height))

    # Display the processed frame without saving
    cv2.imshow('Processed Frame', resized_edges)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()