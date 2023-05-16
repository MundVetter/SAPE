import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

def SILU(x):
    return x * sigmoid(x)

# Start the search at x = 0
x0 = np.array([0])

res = minimize(SILU, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

print("The global minimum of the SILU activation function occurs at x =", res.x[0])
print("The value of the SILU activation function at this point is y =", res.fun)
