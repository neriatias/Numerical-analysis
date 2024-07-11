'''מגישים:
נתיב לוי 209150879
נריה אטיאס 316118728
'''
# Neria - https://github.com/neriatias/Numerical-analysis/blob/main/ex1/ex1-analyza.py
# Nativ - https://github.com/nativlevi/Numericle-analysis.git
import numpy as np


def is_invertible(matrix):
    return np.linalg.det(matrix) != 0


def inverse_using_elementary(matrix, integer_only=False):
    if not is_invertible(matrix):
        raise ValueError("The matrix is not invertible")

    n = matrix.shape[0]
    # Creating the identity matrix
    I = np.eye(n)
    # Concatenating the original matrix with the identity matrix
    augmented_matrix = np.hstack((matrix, I))

    # Performing elementary matrix inversions
    for i in range(n):
        # Divide the current row to ensure the leading diagonal is 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # Zero out all elements in the current column except the leading diagonal
        for j in range(n):
            if i != j:
                augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[j, i] * augmented_matrix[i]

    # The inverse matrix is the right part of the augmented matrix
    inverse_matrix = augmented_matrix[:, n:]

    if integer_only:
        inverse_matrix = np.round(inverse_matrix).astype(int)

    return inverse_matrix


def max_row_norm(matrix):
    # Compute the sum of absolute values in each row
    row_sums = np.sum(np.abs(matrix), axis=1)
    # Find the maximum sum among the row sums
    max_norm = np.max(row_sums)
    return max_norm


# Example of a 3x3 matrix
A = np.array([
    [1, -1, -2],
    [2, -3, -5],
    [-1, 3, 5]
])

# Compute the maximum row norm for the original matrix
original_max_norm = max_row_norm(A)
print("The maximum row norm of the original matrix:", original_max_norm)

# Compute the inverse matrix and its maximum row norm
try:
    inverse_matrix = inverse_using_elementary(A, integer_only=True)
    print("The inverse matrix:")
    print(inverse_matrix)

    inverse_max_norm = max_row_norm(inverse_matrix)
    print("The maximum row norm of the inverse matrix:", inverse_max_norm)

    # Compute the product of the norms and store it in the COND variable
    COND = original_max_norm * inverse_max_norm
    print("The value of COND:", COND)
except ValueError as e:
    print(e)
