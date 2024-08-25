import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def homomorphic_filter(image, low_gain=0.5, high_gain=2.0, cutoff=30):
    # Logarithmic transformation
    image_log = np.log1p(np.array(image, dtype="float"))
    
    # Fourier transform
    image_fft = fft2(image_log)
    image_fft_shift = fftshift(image_fft)
    
    # Create high-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = (high_gain - low_gain) * (1 - np.exp(- (D**2) / (2 * (cutoff**2)))) + low_gain
    
    # Apply high-pass filter
    image_fft_filt = H * image_fft_shift
    image_fft_filt_shift = ifftshift(image_fft_filt)
    
    # Inverse Fourier transform
    image_filt = ifft2(image_fft_filt_shift)
    image_filt = np.exp(np.real(image_filt)) - 1
    
    # Normalize to 8-bit image
    image_filt = np.uint8(cv2.normalize(image_filt, None, 0, 255, cv2.NORM_MINMAX))
    
    return image_filt

# Load an example image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Apply homomorphic filter
filtered_image = homomorphic_filter(image)

# Display the original and filtered images
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(filtered_image, cmap='gray')
ax[1].set_title('Filtered Image (Homomorphic Filtering)')
ax[1].axis('off')

plt.show()
