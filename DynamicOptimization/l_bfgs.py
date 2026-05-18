
import time
import numpy as np
from collections import deque

def l_bfgs(f, grad_f, x0, max_duration = None, max_steps = None, m: int = 5, alpha0: float = 1.0, c1: float = 1e-4, 
           rho: float = 0.5, epsilon: float = 1e-4):
    if (max_duration is None and max_steps is None):
        raise ValueError("You have to set a maximum duration or maximum number of steps to avoid infinite loops")

    xk = x0
    memory = deque(maxlen=m)
    steps = 0
    start = time.time()
    
    while True:
        q = grad_f(xk)

        for i in range(len(memory)):
            alphai = memory[-i-1]["rhok"] * (memory[-i-1]["sk"].T @ q)
            q -= alphai * memory[-i-1]["yk"]
            memory[-i-1]["alpha"] = alphai
        
        if steps == 0:
            r = q
        else:
            phik = (memory[-1]["yk"].T @ memory[-1]["sk"]) / (memory[-1]["yk"].T @ memory[-1]["yk"])
            r = phik * q

        for i in range(len(memory)):
            beta = memory[i]["rhok"] * (memory[i]["yk"].T @ r)
            r += memory[i]["sk"]*(memory[i]["alpha"] - beta)
        
        pk = -r

        alpha = alpha0
        while (f(xk + alpha*pk) > f(xk) + c1*alpha * grad_f(xk).T @ pk):
            alpha *= rho
        xk_1 = xk + alpha*pk
        steps += 1

        sk = xk_1 - xk
        yk = grad_f(xk_1) - grad_f(xk)
        xk = xk_1      

        if np.linalg.norm(grad_f(xk_1)) <= epsilon:
            break
        if max_duration is not None and (time.time() - start >= max_duration):
            break
        if max_steps is not None and steps == max_steps:
            break 
        
        if(yk.T@sk > 0):
            rhok = 1 / (yk.T @ sk)
            if len(memory) == m:
                memory.popleft()
            memory.append({"sk": sk, "yk": yk, "rhok": rhok})

    return xk