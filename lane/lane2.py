import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")

# Set the desired output video width and height
output_width, output_height = 640, 360

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Apply GaussianBlur to the frame
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame, 50, 150)

    # Define region of interest (ROI) for lane detection
    height, width = frame.shape[:2]
    trapezium_vertices = np.array([[(100, height), (width // 2 - 50, height // 2 + 50),
                                    (width // 2 + 50, height // 2 + 50), (width - 100, height)]], dtype=np.int32)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, [trapezium_vertices], 255)
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    # Apply Hough line transformation
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw a trapezium around the detected lane area
    cv2.polylines(frame, [trapezium_vertices], isClosed=True, color=(0, 255, 0), thickness=2)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Resize the frame before displaying
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Display the processed frame
    cv2.imshow("Lane Detection", resized_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
