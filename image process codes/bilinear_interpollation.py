import numpy as np
import cv2

def bilinear_interpolation(image, zoom_factor):
    original_height, original_width = image.shape[:2]
    new_height = int(original_height * zoom_factor)
    new_width = int(original_width * zoom_factor)
    
    new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            x = i / zoom_factor
            y = j / zoom_factor
            
            x1 = int(np.floor(x))
            x2 = min(x1 + 1, original_height - 1)
            y1 = int(np.floor(y))
            y2 = min(y1 + 1, original_width - 1)
            
            dx = x - x1
            dy = y - y1
            
            for c in range(image.shape[2]):
                I_y1 = (1 - dx) * image[x1, y1, c] + dx * image[x2, y1, c]
                I_y2 = (1 - dx) * image[x1, y2, c] + dx * image[x2, y2, c]
                new_image[i, j, c] = int((1 - dy) * I_y1 + dy * I_y2)
    
    return new_image

# Load an image using OpenCV. you should use your own
image = cv2.imread('input_image.jpg')

# Set the zoom factor
zoom_factor = 2.0

# Perform bilinear interpolation
zoomed_image = bilinear_interpolation(image, zoom_factor)

# Save or display the zoomed image
cv2.imwrite('zoomed_image.jpg', zoomed_image)
cv2.imshow('Zoomed Image', zoomed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
