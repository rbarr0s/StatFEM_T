import numpy as np
from scipy.linalg import cho_factor, cho_solve
from gp_model import squared_exponential_covariance

def compute_log_likelihood(Y, P, u_prior, Cu_prior, sensor_locations, sigma_e, rho, sigma_d, l_d):
    """
    Compute the total log-likelihood for the observation matrix Y given hyperparameters.
    
    The model is:
        y = ρ P u + d + e,
    with
        u ~ N(u_prior, Cu_prior),
        d ~ GP(0, K_d) with K_d computed using (σ_d, l_d),
        e ~ N(0, σ_e² I).
    
    After marginalizing u, one obtains:
        y ~ N(ρ P u_prior, Q), 
    with:
        Q = ρ² P Cu_prior Pᵀ + K_d + σ_e² I.
    
    :param Y: Observation matrix (ny x no), each column is an observation vector.
    :param P: Projection matrix (ny x n_u).
    :param u_prior: Finite element prior mean (n_u,).
    :param Cu_prior: Finite element prior covariance (n_u x n_u).
    :param sensor_locations: NumPy array of sensor locations (ny,).
    :param sigma_e: Observation noise standard deviation.
    :param rho: Scaling hyperparameter.
    :param sigma_d: Model discrepancy scale hyperparameter.
    :param l_d: Model discrepancy lengthscale hyperparameter.
    :return: Total log-likelihood (scalar).
    """
    ny, no = Y.shape
    m = rho * (P @ u_prior)  # mean vector (ny,)
    
    # Covariance contribution from FE prior:
    C_fe = rho**2 * (P @ Cu_prior @ P.T)
    
    # Covariance from model inadequacy d:
    K_d = squared_exponential_covariance(sensor_locations, sigma_d, l_d)
    
    # Observation noise covariance:
    C_e = sigma_e**2 * np.eye(ny)
    
    # Total covariance:
    Q = C_fe + K_d + C_e
    
    # Cholesky factorization for numerical stability:
    try:
        L, lower = cho_factor(Q, lower=True)
    except np.linalg.LinAlgError as e:
        print("Cholesky decomposition failed:", e)
        return -np.inf
    
    log_det_Q = 2 * np.sum(np.log(np.diag(L)))
    log_likelihood_total = 0.0
    
    # Compute log-likelihood for each observation vector (each column of Y)
    for i in range(no):
        y = Y[:, i]
        diff = y - m
        sol = cho_solve((L, lower), diff)  # Q⁻¹ (y - m)
        quad_form = diff.T @ sol  # (y - m)ᵀ Q⁻¹ (y - m)
        log_likelihood = -0.5 * (quad_form + log_det_Q + ny * np.log(2 * np.pi))
        log_likelihood_total += log_likelihood
    
    return log_likelihood_total
