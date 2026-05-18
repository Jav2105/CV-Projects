
import time
import numpy as np

def damped_newton(f, grad_f, hessian_f, x0, max_duration = None, max_steps = None, alpha0: float = 1.0, c1: float = 1e-4, 
                  rho: float = 0.5, epsilon: float = 1e-4):
    if (max_duration is None and max_steps is None):
        raise ValueError("You have to set a maximum duration or maximum number of steps to avoid infinite loops")
    
    x = x0
    steps = 0
    start = time.time()
    while True:
        pk = np.linalg.solve(hessian_f(x), -grad_f(x))
        alpha = alpha0
        while (f(x+alpha*pk)> f(x) + c1*alpha * grad_f(x).T @ pk):
            alpha *= rho
        x += alpha*pk
        steps += 1

        if np.linalg.norm(grad_f(x)) <= epsilon:
            break
        if max_duration is not None and (time.time() - start >= max_duration):
            break
        if max_steps is not None and steps == max_steps:
            break
    return x