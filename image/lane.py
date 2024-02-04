import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained lane detection model
model = load_model('image/50.h5')

# Set the desired display width and height
display_width = 640
display_height = 360


# Function to process each frame and detect lanes
def detect_lanes(frame):
    # Preprocess the frame (resize, normalize, etc.)
    processed_frame = preprocess_frame(frame)

    # Make predictions using the lane detection model
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))[0]

    # Post-process the predictions (thresholding, filtering, etc.)
    processed_predictions = postprocess_predictions(predictions)

    # Overlay the detected lanes on the original frame
    result_frame = overlay_lanes(frame, processed_predictions)

    return result_frame

# Function to preprocess the frame before making predictions
def preprocess_frame(frame):
    # Resize the frame to the input size expected by the model
    processed_frame = cv2.resize(frame, (160, 80))
    processed_frame = processed_frame / 255.0  # Normalize pixel values to [0, 1]
    return processed_frame

# Function to post-process the model predictions
def postprocess_predictions(predictions):
    # Implement any necessary post-processing (thresholding, filtering, etc.)
    # Example: threshold the predictions to get binary mask
    threshold = 0.5
    processed_predictions = (predictions > threshold).astype(np.uint8)
    return processed_predictions

# Function to overlay detected lanes on the original frame
def overlay_lanes(frame, predictions):
    # Resize predictions to match the shape of the original frame
    predictions_resized = cv2.resize(predictions, (frame.shape[1], frame.shape[0]))

    # Create a color mask based on predictions
    color_mask = np.zeros_like(frame)
    color_mask[:, :, 0] = predictions_resized * 255  # Red channel

    # Overlay the color mask on the original frame
    result_frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

    return result_frame


# Open video capture
cap = cv2.VideoCapture('image/testvideo3.mp4')  # Replace 'your_video.mp4' with your video file

# Get the frame dimensions
width = int(cap.get(3))
height = int(cap.get(4))

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (width, height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detect lanes in the current frame
    result_frame = detect_lanes(frame)

    # Resize the frame for display
    resized_frame = cv2.resize(result_frame, (display_width, display_height))

    # Display the result frame
    cv2.imshow('Lane Detection', resized_frame)

    # Save the result frame to the output video
    out.write(result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
