import cv2
import numpy as np

def process_frame(frame):
    # Step 1: Convert the frame to black and white
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Apply region of interest mask
    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, region_of_interest_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Step 5: Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Step 6: Draw the detected lines on the original frame
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # Combine the original frame with the line image
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return result

# Open the video file
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow("Lane Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
