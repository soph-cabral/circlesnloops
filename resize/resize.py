import cv2
import os

# Function to rotate, resize, and center crop the image
def rotate_resize_and_center_crop(image, canvas_size):
    # Rotate the image 90 degrees clockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    original_height, original_width = rotated_image.shape[:2]
    canvas_width, canvas_height = canvas_size

    # Calculate aspect ratios
    aspect_ratio_original = original_width / original_height
    aspect_ratio_canvas = canvas_width / canvas_height

    # Resize the image while maintaining aspect ratio
    if aspect_ratio_original > aspect_ratio_canvas:
        # Resize based on height (width will be larger than canvas, crop width)
        new_height = canvas_height
        new_width = int(canvas_height * aspect_ratio_original)
    else:
        # Resize based on width (height will be larger than canvas, crop height)
        new_width = canvas_width
        new_height = int(canvas_width / aspect_ratio_original)

    # Resize the image
    resized_image = cv2.resize(rotated_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate the cropping points (to center crop)
    x_offset = (new_width - canvas_width) // 2
    y_offset = (new_height - canvas_height) // 2

    # Crop the image to the canvas size
    cropped_image = resized_image[y_offset:y_offset + canvas_height, x_offset:x_offset + canvas_width]

    return cropped_image

# Specify the folder path for input images
folder_path = 'image_to_resize'

# Specify the output folder path for resized images
output_folder = 'resized_image'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a counter for sequential naming
counter = 0

# Process each image in the input folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is not None:
            print(f"Loaded image from {filename}")

            # Define the canvas size (target size)
            canvas_size = (512, 512)

            # Rotate, resize, and center crop the image
            cropped_image = rotate_resize_and_center_crop(image, canvas_size)

            # Generate the new filename in the format image_XX.jpg
            new_filename = f"image_{counter:02d}.jpg"
            output_path = os.path.join(output_folder, new_filename)

            # Save the processed image with the new name
            cv2.imwrite(output_path, cropped_image)
            print(f"Processed image saved as {output_path}")

            # Increment the counter for the next image
            counter += 1
        else:
            print(f"Error: Could not load image {filename}")
    else:
        continue
