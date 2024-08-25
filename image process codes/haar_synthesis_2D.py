import numpy as np

def haar_synthesis(LL, LH, HL, HH):
    rows, cols = LL.shape
    reconstructed_image = np.zeros((rows * 2, cols * 2))

    for i in range(rows):
        for j in range(cols):
            reconstructed_image[2*i, 2*j] = (LL[i, j] + LH[i, j] + HL[i, j] + HH[i, j]) / 2
            reconstructed_image[2*i+1, 2*j] = (LL[i, j] - LH[i, j] + HL[i, j] - HH[i, j]) / 2
            reconstructed_image[2*i, 2*j+1] = (LL[i, j] + LH[i, j] - HL[i, j] - HH[i, j]) / 2
            reconstructed_image[2*i+1, 2*j+1] = (LL[i, j] - LH[i, j] - HL[i, j] + HH[i, j]) / 2

    return reconstructed_image

# Example usage with random subbands
LL = np.random.rand(4, 4)
LH = np.random.rand(4, 4)
HL = np.random.rand(4, 4)
HH = np.random.rand(4, 4)

reconstructed_image = haar_synthesis(LL, LH, HL, HH)
print(reconstructed_image)
