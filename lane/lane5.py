import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=(255, 0, 0), thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def detect_lanes(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (trapezium shape)
    height, width = img.shape[:2]
    roi_vertices = np.array([[(100, height), (width // 2 - 50, height // 2 + 50),
                              (width // 2 + 50, height // 2 + 50), (width - 100, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, roi_vertices)

    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Check if lines were detected
    if lines is not None:
        # Draw the lines on the original image
        line_img = np.zeros_like(img)
        draw_lines(line_img, lines)
        return line_img
    else:
        # If no lines detected, return the original image
        return img


# Example usage
cap = cv2.VideoCapture('video.mp4')  # Replace with your video file
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lane_img = detect_lanes(frame)

    # Display the result
    cv2.imshow('Lane Detection', lane_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
