import numpy as np

def haar_wavelet_synthesis(LL, LH, HL, HH):
    """
    Perform the Haar wavelet synthesis (inverse DWT) to reconstruct an image from its subbands.
    
    Parameters:
    LL : 2D numpy array
        The approximation coefficients.
    LH : 2D numpy array
        The horizontal detail coefficients.
    HL : 2D numpy array
        The vertical detail coefficients.
    HH : 2D numpy array
        The diagonal detail coefficients.
        
    Returns:
    reconstructed_image : 2D numpy array
        The reconstructed image.
    """
    rows, cols = LL.shape
    reconstructed_image = np.zeros((rows * 2, cols * 2))
    
    for i in range(rows):
        for j in range(cols):
            reconstructed_image[2 * i, 2 * j] = (LL[i, j] + LH[i, j] + HL[i, j] + HH[i, j]) / 2
            reconstructed_image[2 * i + 1, 2 * j] = (LL[i, j] - LH[i, j] + HL[i, j] - HH[i, j]) / 2
            reconstructed_image[2 * i, 2 * j + 1] = (LL[i, j] + LH[i, j] - HL[i, j] - HH[i, j]) / 2
            reconstructed_image[2 * i + 1, 2 * j + 1] = (LL[i, j] - LH[i, j] - HL[i, j] + HH[i, j]) / 2
    
    return reconstructed_image

# Example usage with random subbands
LL = np.random.rand(4, 4)
LH = np.random.rand(4, 4)
HL = np.random.rand(4, 4)
HH = np.random.rand(4, 4)

reconstructed_image = haar_wavelet_synthesis(LL, LH, HL, HH)
print(reconstructed_image)
