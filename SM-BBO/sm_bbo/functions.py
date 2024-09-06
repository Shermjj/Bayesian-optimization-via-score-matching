import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction

class Rosenbrock(SyntheticTestFunction):
    def __init__(self, 
                 dim: int = 2,
                 noisy: bool = False):
        self.dim = dim
        self._bounds = [(-5.0, 10.0)] * self.dim
        self._optimal_value = 0.0
        self._optimizers = [(1.0,)] * self.dim
        if noisy: 
            self._fn = add_noise(rosenbrock_function)
        else: 
            self._fn = rosenbrock_function
        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.tensor(
            np.apply_along_axis(self._fn,
                                1,
                                X.numpy()
                                )
        )
        
class Rastrigin(SyntheticTestFunction):
    def __init__(self, 
                 dim: int = 2,
                 noisy: bool = False):
        self.dim = dim
        self._bounds = [(-5.12, 5.12)] * self.dim
        self._optimal_value = 0.0
        self._optimizers = [(0.0,)] * self.dim
        if noisy: 
            self._fn = add_noise(rastrigin_function)
        else: 
            self._fn = rastrigin_function
        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.tensor(
            np.apply_along_axis(self._fn,
                                1,
                                X.numpy()
                                )
        )

class Ackley(SyntheticTestFunction):
    def __init__(self, 
                 dim: int = 2,
                 noisy: bool = False):
        self.dim = dim
        self._bounds = [(-32.768, 32.768)] * self.dim
        self._optimal_value = 0.0
        self._optimizers = [(0.0,)] * self.dim
        if noisy: 
            self._fn = add_noise(ackley_function)
        else: 
            self._fn = ackley_function
        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.tensor(
            np.apply_along_axis(self._fn,
                                1,
                                X.numpy()
                                )
        )
        
def add_noise(func):
    def noisy_func(x, noise_level=0.1, *args, **kwargs):
        original_output = func(x, *args, **kwargs)
        noise = np.random.normal(0, noise_level, size=np.shape(original_output))
        return original_output + noise
    
    return noisy_func

def make_max(func):
    # Make the function to be maximized (as is the case for Bayesian Optimization problems)
    def max_func(x, *args, **kwargs):
        return -1 * func(x, *args, **kwargs)
    return max_func

def rastrigin_function(x):
    """https://www.sfu.ca/~ssurjano/rastr.html
    """
    x = np.asarray(x)
    
    d = x.shape[-1]
    
    return 10 * d + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum()

def rosenbrock_function(x, a=1, b=100):
    """
    https://www.sfu.ca/~ssurjano/rosen.html
    """
    x = np.asarray(x)
    
    return np.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)


def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    """
    Ref : https://www.sfu.ca/~ssurjano/ackley.html
    """
    x = np.asarray(x)
    d = x.shape[-1]  # allows for both 1-D and 2-D inputs
    
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=-1) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x), axis=-1) / d)
    
    return sum_sq_term + cos_term + a + np.exp(1)
