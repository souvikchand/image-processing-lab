import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolve2d(image, kernel):
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the dimensions of the output image
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize the output image
    output_image = np.zeros((output_height, output_width))
    
    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            output_image[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output_image

# Load an example image in grayscale
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Define a Sobel kernel for edge detection
sobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Perform the convolution
convolved_image = convolve2d(image, sobel_kernel)

# Display the original and convolved images
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(convolved_image, cmap='gray')
ax[1].set_title('Convolved Image (Edge Detection)')
ax[1].axis('off')

plt.show()
