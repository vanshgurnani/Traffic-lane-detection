import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('road.mp4')

# Set the desired width and height for resizing
desired_width = 640
desired_height = 480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = np.array([[0, height], [width / 2, height / 2], [width, height]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=50)

    # Draw lines on the original frame
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combine the original frame with the line image
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    cv2.imshow('Lane Detection', result)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
