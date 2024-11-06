import cv2
import os
import numpy as np

# Define the input folder and output folder
input_folder = 'images/bottom_floor'
output_folder = 'scaled/bottom_floor'

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Scaling factor (20% downscale means 80% of the original size)
scale_factor = 0.3

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add more extensions if needed
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {filename}")
            continue

        # Get the dimensions of the original image
        height, width = image.shape[:2]

        # Calculate new size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a black canvas of the original size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate the top-left corner to place the scaled image (center it)
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2

        # Place the scaled image onto the black canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = scaled_image

        # Save the result to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, canvas)

        print(f"Processed and saved: {output_path}")

print("Processing complete!")
