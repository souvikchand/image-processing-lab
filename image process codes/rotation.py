import cv2
import matplotlib.pyplot as plt

# Load an image (grayscale for simplicity)
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Rotate the image by pi/4 radians (45 degrees)
rows, cols = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Calculate histograms
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_rotated = cv2.calcHist([rotated_image], [0], None, [256], [0, 256])

# Plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(hist_original, color='b')
plt.title('Original Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')

plt.subplot(122)
plt.plot(hist_rotated, color='r')
plt.title('Rotated Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')

plt.tight_layout()
plt.show()
