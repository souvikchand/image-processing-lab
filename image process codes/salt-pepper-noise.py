import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generate salt-and-pepper noise in an image
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    salt_noise = np.random.rand(*image.shape) < salt_prob
    pepper_noise = np.random.rand(*image.shape) < pepper_prob
    noisy_image[salt_noise] = 255
    noisy_image[pepper_noise] = 0
    return noisy_image


# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Add salt-and-pepper noise to the image
noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)

# Apply median filter to remove noise
denoised_image = cv2.medianBlur(noisy_image, 3)  # 3x3 median filter kernel

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Denoised Image')
plt.imshow(denoised_image, cmap='gray')

plt.show()
