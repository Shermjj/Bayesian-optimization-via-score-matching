import numpy as np
import optax

def fit_BO(param, 
           D,
           target_fn,
           max_fn_tau,
           var_hyper,
           optimizer,
           T,
           K,
           M):

    for _ in range(T):
        param, D = fit_acqui_fn(param,
                                D, 
                                target_fn, 
                                max_fn_tau,
                                optimizer,
                                M, 
                                K,
                                var_hyper)
        # K * M function evaluations
        # Do one function evaluation for param, update the dataset
        theta_samples, func_evals = simulate_samples_and_grad(target_fn, param, 1, var_hyper)
        D.extend(list(zip(theta_samples, func_evals)))

        param, max_fn_tau = max(D, key=lambda x: x[1])
        # Greedy update for param

    return D

def fit_acqui_fn(param, 
                D,
                target_fn,
                max_fn_tau,
                optimizer,
                M, 
                K,
                var_hyper):
    opt_state = optimizer.init(param)

    def anneal_var_hyper(var_hyper, t, T):
        return var_hyper * max(0, 1 - ((t+0.9) / T))  

    def step(param, opt_state):
        # Do the sampling step
        theta_samples, func_evals = simulate_samples_and_grad(target_fn,
                                                              param,
                                                              M,
                                                              var_hyper) 

        D.extend(list(zip(theta_samples, func_evals)))
        
        # Thresholding
        func_evals_thresholded = func_evals > max_fn_tau
        
        # Proposal Grads
        grads_q = batch_neg_grad_log_q(theta_samples,
                                       param,
                                       var_hyper)
        
        # Gradient Calculation
        # Note, this is a negative gradient, as we are maximizing the acquisition function
        grads = - est_sm_gradient(func_evals_thresholded,
                                grads_q)

        updates, opt_state = optimizer.update(grads, opt_state)
        param = optax.apply_updates(param, updates)

        return param, opt_state, grads, func_evals_thresholded.mean()

    for k in range(K):
        param, opt_state, _, acceptance_rate = step(param, opt_state)
        var_hyper = anneal_var_hyper(var_hyper, k, K)
    
    return param, D

def neg_grad_log_q(x, mu, var):
    return np.linalg.inv(var) @ (x - mu)

def batch_neg_grad_log_q(x, mu, var):
    """x is a (batch_dim, theta_dim) input
    """
    grads_q = np.einsum(
        'ij,kj->ki', 
        np.linalg.inv(var),
        x - mu)

    # (M, param_dim) tensor
    return grads_q 

def simulate_samples_and_grad(target_fn,
                     param,
                     M,
                     prop_var):
    """
    Simulate samples around the param, and return the parameter simulation and the corresponding function value simulations (thresholded)
    """
    theta_samples = np.random.multivariate_normal(
        param,
        np.sqrt(prop_var),
        M)
    # (M, param_dim) tensor
    
    func_evals = np.apply_along_axis(
        target_fn, 
        1, 
        theta_samples)
    # (M, ) tensor
    

    return theta_samples, func_evals

def est_sm_gradient(thresholded_func_evals,
                    grads_q):
    
    """
    Returns the gradient for the local 
    """
    
    Z = sum(thresholded_func_evals)
    M = thresholded_func_evals.shape[0]
    theta_dim = grads_q.shape[-1]
    thresholded_grads_q = grads_q[thresholded_func_evals, :]
    
    if Z == 0: return np.zeros(theta_dim)
    
    return thresholded_grads_q.mean(0) / (Z / M)