
import time
import numpy as np

def bfgs(f, grad_f, x0, max_duration = None, max_steps = None, alpha0: float = 1.0, c1: float = 1e-4, rho: float = 0.5, 
         epsilon: float = 1e-4):
    if (max_duration is None and max_steps is None):
        raise ValueError("You have to set a maximum duration or maximum number of steps to avoid infinite loops")
    
    B = np.eye(len(grad_f(x0)))
    xk = x0
    steps = 0
    start = time.time()
    while True:
        pk = np.linalg.solve(B, -grad_f(xk))
        alpha = alpha0
        # Armijo and curvature conditions
        while (f(xk + alpha*pk) > f(xk) + c1*alpha * grad_f(xk).T @ pk):
            alpha *= rho
        xk_1 = xk + alpha*pk
        steps += 1

        sk = xk_1 - xk
        yk = grad_f(xk_1) - grad_f(xk)
        if(yk.T@sk > 0):
            B += - (B @ sk @ sk.T @ B)/(sk.T @ B @ sk) + (yk @ yk.T)/(yk.T @ sk)
        xk = xk_1

        if np.linalg.norm(grad_f(xk_1)) <= epsilon:
            break
        if max_duration is not None and (time.time() - start >= max_duration):
            break
        if max_steps is not None and steps == max_steps:
            break

    return xk