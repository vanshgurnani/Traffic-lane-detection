import cv2
import numpy as np
import csv

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Read image
img = cv2.imread("image.jpeg")
height, width, _ = img.shape

# Preprocess image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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

# Implementing Non-Maximum Suppression (NMS)
if len(boxes) > 0:
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indices) > 0:
        # Drawing green boxes around detected objects on the image (after NMS)
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            # Draw a green rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color: (0, 255, 0), Thickness: 2

        # Save the image with green boxes drawn around detected objects (after NMS)
        cv2.imwrite("output_image_with_nms.jpg", img)
        print("Image with green boxes around detected objects (after NMS) saved as 'output_image_with_nms.jpg'")

        # Count the number of detected objects after NMS
        num_objects_after_nms = len(indices)
        print(f"Number of objects detected after applying NMS: {num_objects_after_nms}")
        
        
        # Initialize CSV file for NMS results
        csv_nms_results = "nms_results.csv"

        # Open CSV file in write mode and create a CSV writer object for NMS results
        with open(csv_nms_results, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write header row for NMS results
            writer.writerow(['Item Name', 'Class ID', 'Confidence', 'X', 'Y', 'Width', 'Height'])

            # Drawing green boxes around detected objects on the image (after NMS)
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                item_name = classes[class_id]

                # Write information of each selected bounding box to the CSV file
                writer.writerow([item_name, class_id, confidence, x, y, w, h])

        print(f"Selected detected objects' information after NMS saved in '{csv_nms_results}'")
        
        
    else:
        print("No valid indices found after applying NMS.")
else:
    print("No boxes detected to apply NMS.")
    
    
# Save the image with green boxes drawn around detected objects (after NMS)
cv2.imwrite("output_image_with_nms.jpg", img)

print("Image with green boxes around detected objects (after NMS) saved as 'output_image_with_nms.jpg'")