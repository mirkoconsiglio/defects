import numpy as np

# Example matrix
matrix = np.array([[1.0, 0.5], [0.5, 2.0]])

# Save the matrix as a .npy file
np.save('matrix.npy', matrix)

# Load it back
loaded_matrix = np.load('matrix.npy')
print(loaded_matrix)