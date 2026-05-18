
from gradient_descent import gradient_descent
from damped_newton import damped_newton
from bfgs import bfgs
from l_bfgs import l_bfgs
import time
import math
import numpy as np

class Descent:
    def __init__(self, f, grad_f, hessian_f, epsilon: float = 1e-4):
        self.f = f
        self.grad_f = grad_f
        self.hessian_f = hessian_f
        self.epsilon = epsilon
        # Time that each model takes to run one iteration
        self.iteration_times = {}
        # Time that each model takes to run until convergence
        self.total_times = {}

    def error(self, x):
        return np.linalg.norm(self.grad_f(x))

    # To find out the processing time for optimum descent
    def calibrate(self, x0, lr: float = 0.01, beta: float = 0.9, alpha0: float = 1.0, c1: float = 1e-4, rho: float = 0.5, 
                    m: int = 5):
        initial_error = self.error(x0)

        start_gd = time.time()
        x1_gd = self.gradient_descent(x0, max_steps=1, lr=lr, beta=beta)
        end_gd = time.time()

        start_newton = time.time()
        x1_newton = self.damped_newton(x0, max_steps=1, alpha0=alpha0, c1=c1, rho=rho)
        end_newton = time.time()
        
        start_bfgs = time.time()
        self.bfgs(x0, max_steps=1)
        end_bfgs = time.time()

        start_l_bfgs = time.time()
        self.l_bfgs(x0, max_steps=1, m=m)
        end_l_bfgs = time.time()
        
        self.iteration_times["gd"] = end_gd - start_gd
        self.iteration_times["newton"] = end_newton - start_newton
        self.iteration_times["bfgs"] = end_bfgs - start_bfgs
        self.iteration_times["l_bfgs"] = end_l_bfgs - start_l_bfgs

        if(self.error(x1_gd)) != 0:
            n_iterations_gd = math.log(self.epsilon/initial_error)/math.log(self.error(x1_gd)/initial_error)
        else:
            n_iterations_gd = 1
        self.total_times["gd"] = n_iterations_gd * self.iteration_times["gd"]

        if(self.error(x1_newton)) != 0:
            c = self.error(x1_newton)/initial_error**2
            n_iterations_newton = math.log2((1+math.log(self.epsilon)/math.log(c))/(1+math.log(initial_error)/math.log(c)))
        else:
            n_iterations_newton = 1
        self.total_times["newton"] = n_iterations_newton * self.iteration_times["newton"]

        n_iterations_bfgs = math.sqrt(n_iterations_gd * n_iterations_newton)
        self.total_times["bfgs"] = n_iterations_bfgs * self.iteration_times["bfgs"]
        self.total_times["l_bfgs"] = math.sqrt(n_iterations_gd * n_iterations_bfgs) * self.iteration_times["l_bfgs"]

    def gradient_descent(self, x0, max_duration = None, max_steps = None, lr: float = 0.01, beta: float = 0.9):
        return gradient_descent(self.grad_f, x0, max_duration, max_steps, lr, beta, self.epsilon)

    def damped_newton(self, x0, max_duration = None, max_steps = None, alpha0: float = 1.0, c1: float = 1e-4, rho: float = 0.5):
        return damped_newton(self.f, self.grad_f, self.hessian_f, x0, max_duration, max_steps, alpha0, c1, rho, self.epsilon)
    
    def bfgs(self, x0, max_duration = None, max_steps = None, alpha0: float = 1.0, c1: float = 1e-4, rho: float = 0.5):
        return bfgs(self.f, self.grad_f, x0, max_duration, max_steps, alpha0, c1, rho, self.epsilon)
    
    def l_bfgs(self, x0, max_duration = None, max_steps = None, m: int = 5, alpha0: float = 1.0, c1: float = 1e-4, 
               rho: float = 0.5):
        return l_bfgs(self.f, self.grad_f, x0, max_duration, max_steps, m, alpha0, c1, rho, self.epsilon)

    def optimum_descent(self, x0, max_time: float = 15.0, lr: float = 0.01, beta: float = 0.9, alpha0: float = 1.0, 
                        c1: float = 1e-4, rho: float = 0.5, m: int = 5):
        # Hierarchy: Damped Newton, BFGS, L-BFGS, Gradient Descent
        x = x0
        start = time.time()
        if (self.total_times["newton"] <= max_time) or (self.total_times["newton"] == min(self.total_times.values())):
            print("Damped Newton is used exclusively")
            while True:
                # Check that the Hessian is positive definite
                if(np.all(np.linalg.eigvals(self.hessian_f(x)) > 0)):
                    x = self.damped_newton(x, None, 1, alpha0, c1, rho)
                else:
                    print("BFGS used")
                    x = self.bfgs(x, None, 1, alpha0, c1, rho)
                if time.time() - start >= max_time:
                    break

        elif self.total_times["bfgs"] <= max_time:
            print("BFGS and Damped Newton are used")
            while True:
                if(np.all(np.linalg.eigvals(self.hessian_f(x)) > 0)):
                    x = self.damped_newton(x, None, 1, alpha0, c1, rho)
                else:
                    print("BFGS used")
                    x = self.bfgs(x, None, 1, alpha0, c1, rho)
                if time.time() - start >= max_time - self.total_times["bfgs"]:
                    break
            x = self.bfgs(x, self.total_times["bfgs"], None, alpha0, c1, rho)

        elif self.total_times["l_bfgs"] <= max_time:
            print("L-BFGS and Damped Newton are used")
            while True:
                if(np.all(np.linalg.eigvals(self.hessian_f(x)) > 0)):
                    x = self.damped_newton(x, None, 1, alpha0, c1, rho)
                else:
                    print("BFGS used")
                    x = self.bfgs(x, None, 1, alpha0, c1, rho)
                if time.time() - start >= max_time - self.total_times["l_bfgs"]:
                    break
            x = self.l_bfgs(x, self.total_times["l_bfgs"], None, m, alpha0, c1, rho)

        elif self.total_times["gd"] <= max_time:
            print("Gradient Descent and Damped Newton are used")
            while True:
                if(np.all(np.linalg.eigvals(self.hessian_f(x)) > 0)):
                    x = self.damped_newton(x, None, 1, alpha0, c1, rho)
                else:
                    print("BFGS used")
                    x = self.bfgs(x, None, 1, alpha0, c1, rho)
                if time.time() - start >= max_time - self.total_times["gd"]:
                    break
            x = self.gradient_descent(x, self.total_times["gd"], None, lr, beta)

        elif self.total_times["bfgs"] == min(self.total_times.values()):
            print("BFGS is used exclusively")
            x = self.bfgs(x, max_time, None, alpha0, c1, rho)
        elif self.total_times["l_bfgs"] == min(self.total_times.values()):
            print("L-BFGS is used exclusively")
            x = self.l_bfgs(x, max_time, None, m, alpha0, c1, rho)
        else:
            print("Gradient Descent is used exclusively")
            x = self.gradient_descent(x, max_time, None, lr, beta)

        return x