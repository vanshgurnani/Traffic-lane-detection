import cv2
import numpy as np

# Open a video file or capture device
cap = cv2.VideoCapture('test3.mp4')  # Replace 'your_video.mp4' with the path to your video file

# Check if the video file or capture device is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

# Load YOLO net
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load YOLO classes
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Get output layer names
layer_names = net.getUnconnectedOutLayersNames()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Create a blank canvas to draw lines
    line_canvas = np.zeros_like(frame)

    # Set the color for the lines
    line_color = (0, 255, 0)  # Green color

    # Draw green lines on the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(line_canvas, contours, -1, line_color, 2)

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Create blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the YOLO network
    net.setInput(blob)

    # Run forward pass and get predictions
    detections = net.forward(layer_names)

    # Loop through the detections and draw bounding boxes
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get the coordinates of the bounding box
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the bounding box and label on the frame
                # Draw the bounding box and label on the frame with red color
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red color


    # Overlay the canvas and object detection onto the original frame
    result = cv2.addWeighted(frame, 1, line_canvas, 1, 0)

    # Display the result
    cv2.imshow('Lane Edges and Object Detection', result)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
