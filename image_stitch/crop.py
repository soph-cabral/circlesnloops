import cv2

def center_crop_long_rect(image, target_aspect_ratio):
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate target height and width based on the aspect ratio
    if target_aspect_ratio > 1:  # Wider than tall (landscape)
        target_width = w
        target_height = int(w / target_aspect_ratio)
    else:  # Taller than wide (portrait)
        target_height = h
        target_width = int(h * target_aspect_ratio)
    
    # Ensure the crop size does not exceed the image dimensions
    target_height = min(target_height, h)
    target_width = min(target_width, w)
    
    # Calculate the top-left corner of the crop
    start_x = (w - target_width) // 2
    start_y = (h - target_height) // 2
    
    # Check for potential issues with cropping
    if start_x < 0 or start_y < 0 or target_height <= 0 or target_width <= 0:
        raise ValueError("Invalid crop dimensions. Check the target aspect ratio and image dimensions.")
    
    # Crop the center of the image
    cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
    
    return cropped_image

# Load the image
image = cv2.imread('top_floor/1/stitched_result.jpg')

# Check if the image loaded successfully
if image is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Define your desired aspect ratio (e.g., 16:9 = 16/9)
    target_aspect_ratio = 16 / 6  # Change this to your desired aspect ratio

    # Center crop the image to the desired long rectangular shape
    final_image = center_crop_long_rect(image, target_aspect_ratio)

    # Save or display the final result
    cv2.imwrite('long_rect_cropped_image.jpg', final_image)
    print("Cropped image saved as 'long_rect_cropped_image.jpg'")
