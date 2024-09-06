import numpy as np

def random_search(objective_function, 
                  bounds,
                  n_iterations=1000,
                  n_samples=10,
                  best_value=float('-inf'),
                  best_solution=None):
    """
    Implements the random search algorithm for noisy black box optimisation.

    This is for a maximization problem.

    Args:
        objective_function (callable): The function to be optimized. Should accept a numpy array and return a scalar.
        bounds (numpy.ndarray): An array of shape (n_dimensions, 2) specifying the lower and upper bounds for each dimension.
        n_iterations (int): The number of iterations to run the optimization.
        n_samples (int): The number of random samples to evaluate in each iteration.

    Returns:
        tuple: A tuple containing:
            - best_solution (numpy.ndarray): The best solution found.
            - best_value (float): The value of the objective function at the best solution.
    """
    n_dimensions = bounds.shape[0]
    D = []

    for i in range(n_iterations):
        # Generate random samples within the bounds
        samples = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(n_samples, n_dimensions)
        )

        # Evaluate the objective function for all samples
        values = np.array([objective_function(sample) for sample in samples])

        # Update the best solution if a better one is found
        current_best_idx = np.argmin(values)
        if values[current_best_idx] > best_value:
            best_value = values[current_best_idx]
            best_solution = samples[current_best_idx]

        D.append((best_solution, best_value, n_samples * (i + 1)))

    return best_solution, best_value, D
