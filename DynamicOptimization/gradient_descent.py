
import time
import numpy as np

def gradient_descent(grad_f, x0, max_duration = None, max_steps = None, lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-4):
    if (max_duration is None and max_steps is None):
        raise ValueError("You have to set a maximum duration or maximum number of steps to avoid infinite loops")
    
    x = x0
    momentum = np.zeros(grad_f(x).shape)
    steps = 0
    start = time.time()
    while True:
        x_tilde = x + beta*momentum
        new_momentum = -lr * grad_f(x_tilde) + beta*momentum
        x += new_momentum
        momentum = new_momentum
        steps += 1

        if (np.linalg.norm(grad_f(x)) <= epsilon):
            break
        if max_duration is not None and (time.time() - start >= max_duration):
            break
        if max_steps is not None and steps == max_steps:
            break
    return x