import cv2
import numpy as np
import csv
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open video file
video_path = "testvideo3.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the desired output video width and height
output_width = 640  # You can adjust these values as needed
output_height = 480

# Create output directory to save frames
output_directory = "output_frames"
os.makedirs(output_directory, exist_ok=True)

# Initialize CSV file for results
csv_results = os.path.join(output_directory, "detection_results.csv")
with open(csv_results, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write header row for results
    writer.writerow(['Frame Number', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height'])

    # Initialize object count
    object_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
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

                    # Increment object count
                    object_count += 1

        # Implementing Non-Maximum Suppression (NMS)
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

            if len(indices) > 0:
                # Draw the total count on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                count_text = f"Total Vehicles: {object_count}"
                cv2.putText(frame, count_text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    item_name = classes[class_id]

                    # Write detection results to CSV
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    writer.writerow([frame_number, item_name, confidence, x, y, w, h])

                    # Draw a rectangle around the detected vehicle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display the class label above the detected vehicle
                    label_text = f"{item_name}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Resize the frame before saving and displaying
                resized_frame = cv2.resize(frame, (output_width, output_height))

                # Save the frame as an image in the output directory
                frame_filename = os.path.join(output_directory, f"frame_{frame_number}.jpg")
                cv2.imwrite(frame_filename, resized_frame)

                # Display the resized frame
                cv2.imshow("Output", resized_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture when done
cap.release()
cv2.destroyAllWindows()

# Print the total count of objects
print(f"Total objects detected: {object_count}")
print(f"Detection results saved in '{csv_results}'")
