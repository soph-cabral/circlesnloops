import cv2
import numpy as np

# Load the image
image = cv2.imread('long_rect_cropped_image.jpg')

# Get the dimensions of the image
height, width = image.shape[:2]

# Define the points in the original image
src_points = np.array([[100, 100], [400, 50], [300, 400], [50, 300]], dtype='float32')

# Define the points in the output image
dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

# Calculate the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation
warped_image = cv2.warpPerspective(image, matrix, (width, height))

# Display the original and warped images
cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', warped_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
