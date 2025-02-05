import numpy as np
from sensors import generate_sensor_locations, build_projection_matrix
from mcmc_hyperparams import mcmc_hyperparameters

def run_inference(u_prior, Cu_prior, Y, sigma_e, ny, no,
                  fe_nodes=None, domain=(0, 1),
                  initial_hyperparams=[1.0, 1.0, 1.0],
                  mcmc_iterations=1000, proposal_cov=None, prior_func=None):
    """
    Run Bayesian inference to infer hyperparameters [rho, sigma_d, l_d] using MCMC.
    
    :param u_prior: Finite element prior mean (n_u,).
    :param Cu_prior: Finite element prior covariance (n_u x n_u).
    :param Y: Observation matrix (ny x no).
    :param sigma_e: Observation noise standard deviation.
    :param ny: Number of sensors.
    :param no: Number of observation vectors.
    :param fe_nodes: Finite element node positions (if None, assumed equally spaced over domain).
    :param domain: Tuple (min, max) for the FE domain.
    :param initial_hyperparams: Initial guess for [rho, sigma_d, l_d].
    :param mcmc_iterations: Number of MCMC iterations.
    :param proposal_cov: Proposal covariance matrix for MCMC. If None, a default is set.
    :param prior_func: Function returning log prior of hyperparameters. If None, a uniform prior is assumed.
    :return: Tuple (chain, log_posteriors, sensor_locations, P).
    """
    n_u = len(u_prior)
    if fe_nodes is None:
        fe_nodes = np.linspace(domain[0], domain[1], n_u)
    
    sensor_locations = generate_sensor_locations(ny, domain)
    P = build_projection_matrix(sensor_locations, fe_nodes)
    
    # Set default proposal covariance if not provided.
    if proposal_cov is None:
        proposal_cov = np.diag([6e-5,6e-5,6e-5])
    
    chain, log_posteriors = mcmc_hyperparameters(Y, P, u_prior, Cu_prior, sensor_locations,
                                                 sigma_e, initial_hyperparams,
                                                 mcmc_iterations, proposal_cov, prior_func)
    
    return chain, log_posteriors, sensor_locations, P
