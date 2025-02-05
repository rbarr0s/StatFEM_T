import numpy as np

from sensors import build_projection_matrix
from gp_model import squared_exponential_covariance
from numpy.polynomial.legendre import leggauss

def green_1d_dirichlet(x, xi):
    """
    Green's function for the 1D Poisson problem on [0,1]
    with Dirichlet boundary conditions:
        -u'' = f,  u(0)=u(1)=0.
    """
    if x <= xi:
        return x * (1.0 - xi)
    else:
        return xi * (1.0 - x)

def c_z_squared_exponential(xi, eta, sigma, ls):
    """
    Squared-exponential kernel c_z(xi, eta) = sigma^2 * exp(-0.5*((xi-eta)/ls)^2).
    """
    diff = xi - eta
    return sigma**2 * np.exp(-0.5 * (diff / ls)**2)

def build_greens_covariance(sensor_locations, sigma, lengthscale, true_error = 0, nquad=30):
    """
    Build the covariance matrix K of shape (n_sensors, n_sensors) via
        K[i,j] = ∫∫ g(x_i, xi) * c_z(xi, eta) * g(x_j, eta) dxi deta,
    where g is the Green's function for 1D Poisson with Dirichlet BC,
    and c_z is a squared-exponential in the variables xi, eta.
    
    We use a 2D Gauss–Legendre quadrature on [0,1]^2.
    
    :param sensor_locations: 1D array of sensor positions x_i in [0,1].
    :param sigma: amplitude in the squared-exponential kernel.
    :param lengthscale: lengthscale in the squared-exponential kernel.
    :param nquad: number of quadrature points in each dimension.
    :param nugget: optional small diagonal term to add (e.g. 2.5e-5).
    :return: K, an (n_sensors x n_sensors) covariance matrix.
    """
    n_sensors = len(sensor_locations)
    K_z = np.zeros((n_sensors, n_sensors))

    # 1D Gauss–Legendre points & weights on [-1,1]
    gauss_pts, gauss_wts = leggauss(nquad)
    # transform them to [0,1]
    # mapping t->x: x = 0.5*(t + 1)
    # weight factor is 0.5
    quad_xi  = 0.5*(gauss_pts + 1.0)
    quad_eta = 0.5*(gauss_pts + 1.0)
    w_factor = 0.5

    for i in range(n_sensors):
        x_i = sensor_locations[i]
        for j in range(n_sensors):
            x_j = sensor_locations[j]
            val_ij = 0.0
            
            # 2D quadrature
            for a in range(nquad):
                xi  = quad_xi[a]
                w_xi = gauss_wts[a] * w_factor
                for b in range(nquad):
                    eta  = quad_eta[b]
                    w_eta = gauss_wts[b] * w_factor
                    
                    g_i  = green_1d_dirichlet(x_i, xi)
                    g_j  = green_1d_dirichlet(x_j, eta)
                    cval = c_z_squared_exponential(xi, eta, sigma, lengthscale)
                    
                    val_ij += g_i * cval * g_j * w_xi * w_eta
            
            K_z[i,j] = val_ij
    
    K_y = K_z.copy()
    if true_error > 0.0:
        # If 'true_error' is a standard deviation, add true_error^2 on diagonal
        np.fill_diagonal(K_y, np.diag(K_y) + true_error**2)

    return K_z, K_y

def generate_true_response(sensor_locations, true_mean, true_cov_sigma, true_cov_ls, true_error, no=1):
    """
    Generate the true system response z as a GP at sensor locations and then generate multiple
    observations y by adding independent Gaussian noise.
    
    The GP for the true system response has mean true_mean and covariance given by a kernel function,
    here approximated as a squared exponential kernel (as a surrogate for the Green's function g). 
    
    The observation model is:
        y = z + ε,   where ε ~ N(0, true_error² I)
    
    :param sensor_locations: Array of sensor locations (n_sensors,).
    :param true_mean: GP mean for the true system response. Can be a scalar or a vector of length n_sensors.
    :param true_cov_sigma: Scaling factor (amplitude) for the GP covariance.
    :param true_cov_ls: Lengthscale for the GP covariance.
    :param true_error: Standard deviation of the observation noise added to the true response.
    :param no: Number of observation realizations to generate.
    :return: Tuple (z, Y) where z is the latent true response (n_sensors,) and Y is an array of shape (n_sensors, no)
             where each column is an observation y = z + noise.
    """
    n_sensors = len(sensor_locations)
    # Determine the mean vector mu
    if callable(true_mean):
        mu = np.asarray(true_mean(sensor_locations))
    elif np.isscalar(true_mean):
        mu = true_mean * np.ones(n_sensors)
    else:
        mu = np.asarray(true_mean)
        if mu.shape[0] != n_sensors:
            raise ValueError("Length of true_mean must match number of sensor locations.")
    
    K_z, K_y = build_greens_covariance(sensor_locations, true_cov_sigma, true_cov_ls, true_error, nquad=30)
    
    # Sample the latent true response z from the GP:
    z = np.random.multivariate_normal(mu, K_z, size=no).T
    
    # Generate observations: y = z + noise, with independent noise
    Y = np.random.multivariate_normal(mu, K_y, size=no).T
    
    return z, Y, K_z, K_y

def generate_synthetic_data_fe(u_prior, Cu_prior,
                               rho_true, sigma_d_true, l_d_true,
                               ny=11, no=100,
                               fe_nodes=None, domain=(0,1)):
    """
    Generate synthetic data from a finite-element prior plus a GP discrepancy,
    i.e. z = rho * P u + d, with no additional measurement noise.
    
    :param u_prior: Mean of the FE prior (shape (n_u,)).
    :param Cu_prior: Covariance of the FE prior (shape (n_u, n_u)).
    :param rho_true: True scaling parameter rho.
    :param sigma_d_true: True amplitude for the GP discrepancy.
    :param l_d_true: True lengthscale for the GP discrepancy.
    :param ny: Number of sensors.
    :param no: Number of observation vectors (independent realizations).
    :param fe_nodes: Nodal positions for the FE solution. If None, use linspace(domain, n_u).
    :param domain: Domain (min, max) if fe_nodes is not specified.
    :return: (Y, sensor_locations, P) where
             Y is shape (ny, no),
             sensor_locations is length ny,
             P is (ny, n_u).
    """

    n_u = len(u_prior)
    if fe_nodes is None:
        fe_nodes = np.linspace(domain[0], domain[1], n_u)
    
    # Equally spaced sensor locations
    sensor_locations = np.linspace(domain[0], domain[1], ny)
    
    # Build the projection matrix (ny x n_u)
    P = build_projection_matrix(sensor_locations, fe_nodes)
    
    # Precompute the GP discrepancy covariance at sensors
    K_d = squared_exponential_covariance(sensor_locations, sigma_d_true, l_d_true)
    
    # For each "reading", sample u from the FE prior and d from the GP discrepancy
    Y = np.zeros((ny, no))
    for i in range(no):
        # Sample a realization of u
        u_sample = np.random.multivariate_normal(u_prior, Cu_prior)
        
        # Sample a realization of discrepancy d
        d_sample = np.random.multivariate_normal(np.zeros(ny), K_d)
        
        # True system response at sensors (no measurement noise here)
        z = rho_true * (P @ u_sample) + d_sample
        
        Y[:, i] = z
    
    return Y, sensor_locations, P