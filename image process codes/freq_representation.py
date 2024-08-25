import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Compute magnitude spectrum (logarithmic scale for better visualization)
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# Display the original image and its magnitude spectrum
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(magnitude_spectrum, cmap='gray')
ax[1].set_title('Magnitude Spectrum')
ax[1].axis('off')

plt.show()
