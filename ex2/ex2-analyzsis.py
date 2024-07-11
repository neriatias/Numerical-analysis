'''מגישים:
נתיב לוי 209150879
נריה אטיאס 316118728

https://github.com/nativlevi/Numericle-analysis.git
'''

import numpy as np

# Function to check if the matrix is invertible by calculating its determinant
def is_invertible(matrix):
    return np.linalg.det(matrix) != 0

# Function to find the inverse matrix using elementary matrices
def inverse_using_elementary(matrix, integer_only=False):
    if not is_invertible(matrix):
        raise ValueError("The matrix is not invertible")

    n = matrix.shape[0]
    I = np.eye(n)  # Create the identity matrix
    augmented_matrix = np.hstack((matrix, I))  # Concatenate the original matrix with the identity matrix

    for i in range(n):
        diag_element = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / diag_element  # Normalize the main diagonal

        for j in range(n):
            if i != j:
                row_factor = augmented_matrix[j, i]
                augmented_matrix[j] = augmented_matrix[j] - row_factor * augmented_matrix[i]  # Zero out other columns

    inverse_matrix = augmented_matrix[:, n:]  # The inverse matrix is the right part of the augmented matrix

    if integer_only:
        inverse_matrix = np.round(inverse_matrix).astype(int)

    return inverse_matrix

# Function for LU decomposition of a matrix
def lu_decomposition(matrix):
    n = matrix.shape[0]
    L = np.eye(n)  # Create the identity matrix L
    U = matrix.copy()  # Create a copy of the original matrix for U

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor  # Update the L matrix
            U[j] = U[j] - factor * U[i]  # Update the U matrix

    return L, U

# Function to solve the system of equations LY = b and UX = Y
def solve_lu(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    # Solve LY = b
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve UX = Y
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# Example matrix of size 3x3
A = np.array([
    [1, 4, -3],
    [-2, 1, 5],
    [3, 2, 1]
])

# Example solution vector
b = np.array([1,2,3])

try:
    # Calculate the inverse matrix
    inverse_matrix = inverse_using_elementary(A, integer_only=False)
    print("The inverse matrix:")
    print(inverse_matrix)

    # Perform LU decomposition
    L, U = lu_decomposition(A)
    print("L matrix:")
    print(L)
    print("U matrix:")
    print(U)

    # Solve the system of equations A*x = b
    x = solve_lu(L, U, b)
    print("Solution to the system of equations:")
    print(x)
except ValueError as e:
    print(e)
