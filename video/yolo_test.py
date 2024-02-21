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

detected_objects = []  # List to store bounding boxes of detected objects

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
    
    total_lane_area = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Preprocess frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Get information about detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Initialize a list to store objects
    objects = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            
            # Remove detection for person class (assuming it has class_id 0, modify if necessary)
            if classes[class_id] == "person":
                continue
            
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                objects.append(len(boxes) - 1)

    # Implementing Non-Maximum Suppression (NMS) for object detection
    if len(boxes) > 0:
        confidences = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Store bounding boxes for detected objects
        detected_objects = [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []
        
        total_area_covered=0

        # Show green boxes for all detected objects within the detected lane
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            total_area_covered += w * h

            # Check if the object is within the detected lane
            if any(cv2.pointPolygonTest(np.array([(x1, y1), (x2, y2)]), (x + w / 2, y + h / 2), False) > 0 for line in lines):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate density percentage
        total_frame_area = frame.shape[1] * frame.shape[0]
        density_percentage = (total_area_covered / total_frame_area) * 100

        if density_percentage >= density_threshold:
            light_color = (0, 255, 0)  # Green
            light_text = "Green Light"
        else:
            light_color = (0, 0, 255)  # Red
            light_text = "Red Light"

        # Draw the density text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        density_text = f"Density: {density_percentage:.2f}%"
        light_text = f"Traffic Light: {light_text}"
        cv2.putText(frame, density_text, (10, 60), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, light_text, (10, 90), font, 0.7, light_color, 2, cv2.LINE_AA)

        # Draw count text on the frame
        count_text = f"Total Vehicle Count: {len(detected_objects)}"
        cv2.putText(frame, count_text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw bounding boxes for detected objects
    for box in detected_objects:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Blue color: (255, 0, 0)

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
