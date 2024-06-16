import numpy as np

# Define the matrix \tilde{X}
X_tilde = np.array([
    [1, 2],
    [3, 4]
])

# Step 1: Compute the transpose of \tilde{X}
X_tilde_T = X_tilde.T

# Step 2: Compute \tilde{X}^T \tilde{X}
XtX = np.dot(X_tilde_T, X_tilde)

# Step 3: Compute the inverse of \tilde{X}^T \tilde{X}
XtX_inv = np.linalg.inv(XtX)

# Step 4: Multiply the inverse by \tilde{X}^T
result = np.dot(XtX_inv, X_tilde_T)

# Print the results
print("Transpose of X_tilde (X_tilde_T):\n", X_tilde_T)
print("X_tilde^T X_tilde (XtX):\n", XtX)
print("Inverse of X_tilde^T X_tilde (XtX_inv):\n", XtX_inv)
print("Result (X_tilde^T X_tilde)^-1 X_tilde^T:\n", result)

A = [[1, 2],
              [3, 4]]

# Tính pseudoinverse của A
# A_pinv = np.linalg.pinv(A)
# A_pinv = np.dot(np.invert(np.dot(A.T,A)),np.transpose(A))
A_pinv = np.inverse(np.dot(A.T,A))

# In ma trận A và pseudoinverse của nó
print("Ma trận A:")
print(A)
print("\nPseudoinverse của A:")
print(A_pinv)