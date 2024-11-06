import numpy as np
import cv2 as cv
import os

# Load the previously obtained camera matrix and distortion coefficients
try:
    camera_calibration = np.load('iphone_matrix.npz')
    mtx = camera_calibration['mtx']
    dist = camera_calibration['dist']
except Exception as e:
    print(f"Error loading calibration data: {e}")
    exit()

# Specify the folder containing the images
input_folder = 'scaled/bottom_floor'
output_folder = 'calibrated/bottom_floor'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Process only image files
        # Load the image
        img_path = os.path.join(input_folder, filename)
        img = cv.imread(img_path)

        if img is not None:
            h, w = img.shape[:2]

            # Get the optimal new camera matrix
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # Undistort the image
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)

            # Calculate center coordinates of the undistorted image
            center_x, center_y = dst.shape[1] // 2, dst.shape[0] // 2

            # Define the size of the output image based on ROI
            x_roi, y_roi, w_roi, h_roi = roi
            crop_w = min(w_roi, w)  # Ensure we don't exceed original width
            crop_h = min(h_roi, h)  # Ensure we don't exceed original height

            # Calculate cropping coordinates from the center point
            x_start = center_x - crop_w // 2
            y_start = center_y - crop_h // 2

            # Ensure the cropping coordinates are within the image boundaries
            x_start = max(x_start, 0)
            y_start = max(y_start, 0)
            x_end = min(x_start + crop_w, dst.shape[1])
            y_end = min(y_start + crop_h, dst.shape[0])

            # Crop the image based on the calculated coordinates
            dst_cropped = dst[y_start:y_end, x_start:x_end]

            # Save the undistorted image to the output folder
            output_path = os.path.join(output_folder, f'calibresult_{filename}')
            cv.imwrite(output_path, dst_cropped)

            print(f"Undistorted image saved as {output_path}")
        else:
            print(f"Failed to load image: {filename}")
