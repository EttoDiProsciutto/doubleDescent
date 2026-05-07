import numpy as np

def train_least_squares(X, y):
    """
    Computing ordinary least squares solution.
    """
    # np.linalg.inv inverts matrix (instead of inverting each value)
    w_hat = (np.linalg.inv(X.T@X))@(X.T@y)
    return w_hat

def train_ridge_regression(X, y, alpha):
    """
    Computing Ridge Regression solution.
    Formula: w = (X^T X + alpha * I)^-1 X^T y 
    """
    n, d = X.shape
    # Creating identity matrix dxd o nxn
    # if d>n better to use dual form with identity matrix on n
    if d<n:
        I = np.eye(d)
        w_ridge = (np.linalg.inv(X.T @ X + alpha * I))@(X.T @ y)
    else:
        I = np.eye(n)
        w_ridge = X.T@(np.linalg.inv(X @ X.T + alpha * I))@y    
    
    
    
    return w_ridge

def train_pseudoinverse(X, y):
    """
    Computing least squares solution.
    Using pseudoinverse to achieve the minimal norm solution in over-parameterized (d>n) regime.
    """
    # np.linalg.pinv implementing Moore-Penrose pseudoinversa 
    w_hat = np.linalg.pinv(X) @ y
    return w_hat

def compute_risk(X, y, w, loss_type='squared'):
    """
    Computing model risk.
    Supports 'squared' (MSE) e 'zero-one'.
    """
    y_pred = X @ w
    
    if loss_type == 'squared':
        # Statistical risk for square loss
        return np.mean((y_pred - y)**2)
    
    elif loss_type == 'zero-one':
        # Zero-one loss for classification
        # Assuming y in {-1, 1}
        predictions = np.sign(y_pred)
        return np.mean(predictions != y)
    
def train_gradient_descent(
    X, y,
    lr=1e-3,
    max_iters=2500,
    tol=1e-8
):
    """
    Gradient Descent for linear regression
    with early stopping based on training loss variation.

    Stop when |loss_t - loss_{t-1}| < tol
    """

    n, d = X.shape
    w = np.zeros((d, 1))

    prev_loss = float("inf")
    losses = []

    for t in range(max_iters):

        # --- Gradient step ---
        grad = (2 / n) * X.T @ (X @ w - y) #dividing for n to keep the gradient indipendent from the number of samples
        w = w - lr * grad

        # --- Compute loss ---
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)

        # --- Early stopping ---
        if abs(prev_loss - loss) < tol:
            print(f"Converged at iteration {t}")
            break

        prev_loss = loss

    return w, losses