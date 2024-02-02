import cv2
import numpy as np

# Load the pre-trained ENet model for semantic segmentation
enet = cv2.dnn.readNet("enet-cityscapes/enet-model.net")

# Open video capture
cap = cv2.VideoCapture('image/test3.mp4')  # Replace with your video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the input size expected by the model
    input_blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)

    # Set the input to the model
    enet.setInput(input_blob)

    # Perform a forward pass to get the segmentation mask
    segmentation_mask = enet.forward()

    # Extract lane area coordinates
    area_coordinates_pixel = np.where(segmentation_mask[0, 7] > 0.5)  # Assuming lane class is 7

    # Draw the detected lane area on the original frame
    frame[area_coordinates_pixel] = [0, 255, 0]  # Green color

    # Display the result
    cv2.imshow('Detected Lane', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
