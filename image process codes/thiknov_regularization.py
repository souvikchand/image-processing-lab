import numpy as np
import scipy.signal
import scipy.linalg
import matplotlib.pyplot as plt
from skimage import color, data, restoration

# Load an example image
image = color.rgb2gray(data.camera())

# Simulate a blur kernel (point spread function)
psf = np.ones((5, 5)) / 25
blurred = scipy.signal.convolve2d(image, psf, 'same')

# Add Gaussian noise
np.random.seed(0)
blurred_noisy = blurred + 0.1 * np.random.standard_normal(blurred.shape)

# Display the original, blurred, and blurred + noisy images
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(blurred, cmap='gray')
ax[1].set_title('Blurred Image')
ax[1].axis('off')
ax[2].imshow(blurred_noisy, cmap='gray')
ax[2].set_title('Blurred + Noisy Image')
ax[2].axis('off')
plt.show()

# Define the degradation matrix H (blur kernel)
H = psf

# Define the regularization parameter and matrix
lambda_reg = 0.1
L = np.identity(image.size)

# Perform least squares restoration with Tikhonov regularization
# Reshape image and degradation matrix for matrix operations
g = blurred_noisy.flatten()
H_matrix = scipy.linalg.toeplitz(H.flatten(), np.zeros_like(H.flatten()))

# Regularized least squares solution
H_t_H = H_matrix.T @ H_matrix
lambda_L_t_L = lambda_reg * L.T @ L
H_t_g = H_matrix.T @ g
f_reg = np.linalg.inv(H_t_H + lambda_L_t_L) @ H_t_g

# Reshape the result back to the image format
restored_image = f_reg.reshape(image.shape)

# Display the restored image
plt.imshow(restored_image, cmap='gray')
plt.title('Restored Image')
plt.axis('off')
plt.show()
