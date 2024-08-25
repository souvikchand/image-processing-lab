import numpy as np

# Example 2D transformation matrix for rotation by theta and translation by (tx, ty)
theta = np.pi / 4  # 45 degrees rotation
tx, ty = 2, 3  # Translation by (2, 3)

cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# 2D homogeneous transformation matrix
T = np.array([
    [cos_theta, -sin_theta, tx],
    [sin_theta, cos_theta, ty],
    [0, 0, 1]
])

# Point to be transformed
point = np.array([1, 1, 1])  # Homogeneous coordinates of (1, 1)

# Apply transformation
transformed_point = np.dot(T, point)

print("Transformed point:", transformed_point[:2])  # Discard the homogeneous coordinate
