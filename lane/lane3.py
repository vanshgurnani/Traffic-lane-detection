import cv2
import numpy as np
from keras.models import model_from_json

# Define the Lanes class
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

# Load the pretrained model
json_file = open('model.json', 'r')
json_model = json_file.read()
json_file.close()

model = model_from_json(json_model)
model.load_weights('model.h5')

# Create a Lanes object
lanes = Lanes()

# Define the road_lines_image function
def road_lines_image(frame):
    actual_image = cv2.resize(frame, (1280, 720)).astype(np.uint8)

    small_img_2 = cv2.resize(frame, (160, 80))
    small_img_1 = np.array(small_img_2)
    small_img = small_img_1[None, :, :, :]

    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    lane_image = cv2.resize(lane_drawn, (1280, 720)).astype(np.uint8)
    result = cv2.addWeighted(actual_image, 1, lane_image, 1, 0)

    # Find contours of the green shaded area
    contours, _ = cv2.findContours(lanes.avg_fit.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the bounding box around the green shaded area
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        trapezium_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        cv2.polylines(result, [trapezium_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return result

# Process video using cv2.VideoCapture and display frames with cv2.imshow
def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video_cv2.avi', fourcc, 20.0, (640, 360))  # Reduced size

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = road_lines_image(frame)
        out.write(processed_frame)

        # Resize for display
        display_frame = cv2.resize(processed_frame, (640, 360))
        cv2.imshow('Processed Frame', display_frame)

        # Press 'q' to exit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Specify input video file
input_video_path = 'testvideo3.mp4'

# Process and display the video
process_video(input_video_path)
