import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Detect edges using Canny
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Perform Hough Transform for circle detection
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=10, maxRadius=100)

# Draw circles on the original image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.circle(image, center, 2, (0, 0, 255), 3)

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Hough Transform Circle Detection')
plt.show()
