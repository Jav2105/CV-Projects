
import math
from descent import Descent
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(21)

# Sinusoidal component
def f1(x):
    return x[0][0]*math.cos(x[0][0]) + x[1][0]*math.sin(x[1][0])

def grad_f1(x):
    return np.array([math.cos(x[0][0])-x[0][0]*math.sin(x[0][0]), math.sin(x[1][0])+x[1][0]*math.cos(x[1][0])]).reshape(2,1)

def hessian_f1(x):
    return np.array([[-2*math.sin(x[0][0])-x[0][0]*math.cos(x[0][0]), 0], [0, 2*math.cos(x[1][0])-x[1][0]*math.sin(x[1][0])]])

# Many variables
def f2(x, n):
    result = 0
    for i in range(n):
        result += x[i][0]**2
    return result

def grad_f2(x, n):
    result = np.array([])
    for i in range(n):
        result = np.append(result, 2*x[i][0])
    return result.reshape(n,1)

def hessian_f2(x, n):
    return 2*np.eye(n)

def rosenbrock(x, a, b):
    return (a-x[0][0])**2 + b*(x[1][0] - x[0][0]**2)**2

def grad_rosenbrock(x, a, b):
    return np.array([-2*(a-x[0][0]) - 4*b*x[0][0]*(x[1][0] - x[0][0]**2), 2*b*(x[1][0] - x[0][0]**2)]).reshape(2,1)

def hessian_rosenbrock(x, a, b):
    return np.array([[2 - 4*b*(x[1][0] - 3*x[0][0]**2), -4*b*x[0][0]], [-4*b*x[0][0], 2*b]])

def f3(x):
    return f2(x, 3)

def grad_f3(x):
    return grad_f2(x, 3)

def hessian_f3(x):
    return hessian_f2(x, 3)

F4_DIMENSION = 100
def f4(x):
    return f2(x, F4_DIMENSION)

def grad_f4(x):
    return grad_f2(x, F4_DIMENSION)

def hessian_f4(x):
    return hessian_f2(x, F4_DIMENSION)

def f5(x, n):
    result = 0
    for i in range(n):
        result += x[i][0]*math.exp(x[i][0])
    return result 

def grad_f5(x, n):
    result = np.array([])
    for i in range(n):
        result = np.append(result, (1+x[i][0])*math.exp(x[i][0]))
    return result.reshape(n,1)

def hessian_f5(x, n):
    result = np.array([])
    for i in range(n):
        result = np.append(result, (2+x[i][0])*math.exp(x[i][0]))
    return np.diag(result)

F6_DIMENSION = 100
def f6(x):
    return f5(x, F6_DIMENSION)

def grad_f6(x):
    return grad_f5(x, F6_DIMENSION)

def hessian_f6(x):
    return hessian_f5(x, F6_DIMENSION)

def rosenbrock_1(x):
    return rosenbrock(x, 1, 100)

def grad_rosenbrock_1(x):
    return grad_rosenbrock(x, 1, 100)

def hessian_rosenbrock_1(x):
    return hessian_rosenbrock(x, 1, 100)

# Calibration DOES NOT WORK because of negative values for the number of iterations -> Optimum descent does not work
d1 = Descent(f1, grad_f1, hessian_f1)
x0 = np.array([2.0, 1.0]).reshape(2,1)
#d1.calibrate(x0)
#print(d1.optimum_descent(x0, 10))
'''
print(d1.damped_newton(x0, 15))
print(d1.bfgs(x0, 15))
print(d1.l_bfgs(x0, 15))
print(d1.gradient_descent(x0, 15))
'''

d2 = Descent(f3, grad_f3, hessian_f3)
x0 = np.array([1.0, 2.0, 3.0]).reshape(3,1)
#d2.calibrate(x0)
#print(d2.optimum_descent(x0))
'''
print(d2.damped_newton(x0, 15))
print(d2.bfgs(x0, 15))
print(d2.l_bfgs(x0, 15))
print(d2.gradient_descent(x0, 15))
'''

d3 = Descent(f4, grad_f4, hessian_f4)
x0 = np.random.random(F4_DIMENSION).reshape(F4_DIMENSION,1)
#d3.calibrate(x0)
#d3.optimum_descent(x0, 2.0)
'''
print("F4")
print(d3.damped_newton(x0, 15))
print(d3.bfgs(x0, 15))
print(d3.l_bfgs(x0, 15))
print(d3.gradient_descent(x0, 15))
'''

d4 = Descent(f6, grad_f6, hessian_f6)
x0 = np.random.random(F6_DIMENSION).reshape(F6_DIMENSION,1)
#d4.calibrate(x0)
#d4.optimum_descent(x0, 2.0)
'''
print("F6")
print(d4.damped_newton(x0, 15))
print(d4.bfgs(x0, 15))
print(d4.l_bfgs(x0, 15))
print(d4.gradient_descent(x0, 15))
'''

d5 = Descent(rosenbrock_1, grad_rosenbrock_1, hessian_rosenbrock_1)
x0 = np.array([1.5, 1.5]).reshape(2,1)
#d5.calibrate(x0)
#print(d5.optimum_descent(x0))
'''
print("Rosenbrock")
print("Damped Newton")
print(d5.damped_newton(x0, 15))
print("BFGS")
print(d5.bfgs(x0, 15))
print("L-BFGS")
print(d5.l_bfgs(x0, 15))
print("Gradient descent")
print(d5.gradient_descent(x0, 15))
'''