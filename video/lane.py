import cv2
import numpy as np

# Define a function to convert an image to grayscale
def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a function to apply Gaussian blur
def gaussian_blur(image, kernel_size):
  return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Define a function to apply Canny edge detection
def canny(image, low_threshold, high_threshold):
  return cv2.Canny(image, low_threshold, high_threshold)

# Define a function to create a region of interest (ROI)
def roi(image, height_ratio, width_ratio):
  height, width = image.shape[:2]
  start_y = int(height * height_ratio)
  end_y = height
  start_x = int(width * width_ratio)
  end_x = width
  return image[start_y:end_y, start_x:end_x]

# Define a function to apply Hough transform
def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
  return cv2.HoughLinesP(image, rho, theta, threshold, min_line_length, max_line_gap)

# Read the image
image = cv2.imread("video/image.jpeg")

# Convert the image to grayscale
gray_image = grayscale(image)

# Apply Gaussian blur
blurred_image = gaussian_blur(gray_image, 5)

# Apply Canny edge detection
canny_image = canny(blurred_image, 50, 150)

# Define the region of interest (ROI)
roi_image = roi(canny_image, 0.6, 0.1)

# Apply Hough transform to detect lane lines
lines = hough_lines(roi_image, 2, np.pi/180, 100, 100, 20)

# Draw the detected lane lines on the original image
for line in lines:
  x1, y1, x2, y2 = line[0]
  cv2.line(image, (x1 + int(0.1 * width), y1 + int(0.6 * height)), (x2 + int(0.1 * width), y2 + int(0.6 * height)), (0, 255, 0), 3)

# Show the image with the detected lane lines
cv2.imshow("Image with lane lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
