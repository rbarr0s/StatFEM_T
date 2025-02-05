import numpy as np
import matplotlib.pyplot as plt
from fem_core import statFEM_1d
from inference_runner import run_inference
from sensors import generate_sensor_locations
from data_generator import generate_true_response, generate_synthetic_data_fe, build_greens_covariance
from scipy.linalg import cho_factor, cho_solve

def plot_u_prior_posterior_with_data(
    x_u,             # FE node coordinates (shape (n_u,))
    u_prior_mean,    # prior mean for u (shape (n_u,))
    Cu_prior,        # prior cov  for u (n_u x n_u)
    u_post_mean,     # posterior mean for u (shape (n_u,))
    Cu_post,         # posterior cov  for u (n_u x n_u)
    sensor_locations,# sensor positions (shape (n_y,))
    Y                # observations (shape (n_y, n_o))
):
    """
    1) Plot the prior mean of u (solid line) with 95% CI fill.
    2) Plot the posterior mean of u (solid line) with 95% CI fill.
    3) Scatter all observations as small gray dots at their sensor_locations.

    :param x_u: 1D array of FE node positions, length n_u
    :param u_prior_mean: prior mean array for u, length n_u
    :param Cu_prior: prior covariance (n_u x n_u)
    :param u_post_mean: posterior mean array for u, length n_u
    :param Cu_post: posterior covariance (n_u x n_u)
    :param sensor_locations: positions for sensors (n_y,)
    :param Y: observation matrix, shape (n_y, n_o)
    """
    # ---- Prior 95% intervals
    std_prior = np.sqrt(np.diag(Cu_prior))
    lower_prior = u_prior_mean - 1.96*std_prior
    upper_prior = u_prior_mean + 1.96*std_prior
    
    # ---- Posterior 95% intervals
    std_post = np.sqrt(np.diag(Cu_post))
    lower_post = u_post_mean - 1.96*std_post
    upper_post = u_post_mean + 1.96*std_post

    plt.figure(figsize=(8,5))

    # Plot prior
    plt.plot(x_u, u_prior_mean, color='black', label="u prior mean")
    plt.fill_between(x_u, lower_prior, upper_prior, color='black', alpha=0.2,
                     label="u prior 95% CI")

    # Plot posterior
    plt.plot(x_u, u_post_mean, color='blue', label="u posterior mean")
    plt.fill_between(x_u, lower_post, upper_post, color='blue', alpha=0.2,
                     label="u posterior 95% CI")

    # Plot the data (all columns of Y)
    n_y, n_o = Y.shape
    for i in range(n_o):
        plt.scatter(sensor_locations, Y[:, i], s=5, color='gray', alpha=0.7)
    
    plt.xlabel("x (domain)")
    plt.ylabel("u value")
    plt.title("u Prior vs. Posterior, with observations")
    plt.legend()
    plt.grid(True)
    plt.show()

def posterior_u_given_Y_iid(Y, P, u_prior, Cu_prior, rho, sigma_d, l_d, sigma_e, sensor_locations):
    """
    Computes the posterior for u given n_o independent observations Y,
    using the model:
    
       u ~ N(u_prior, Cu_prior)
       y_i = rho * P * u + d_i + e_i,   for i = 1,..., n_o,
    
    with d_i ~ N(0, K_d) and e_i ~ N(0, sigma_e^2 I), and
         K_d = squared_exponential_covariance(sensor_locations, sigma_d, l_d).
    
    The update equations (Girolami (2021) Eqs. (43a) and (43b)) are:
    
      (a) Q_d = K_d + sigma_e^2 I,
      
      (b) Posterior precision:
          C_{u|Y}^{-1} = C_u^{-1} + n_o * rho^2 * P^T Q_d^{-1} P,
      
      (c) Posterior mean:
          u_{|Y} = C_{u|Y} * ( C_u^{-1} u_prior + rho * P^T Q_d^{-1} * (∑_{i=1}^{n_o} y_i) ).
    
    Here, C_u is Cu_prior and C_{u|Y} is cov_u.
    """
    n_y, n_o = Y.shape  # n_y: number of sensors; n_o: number of observations
    
    print(rho, sigma_d, l_d)

    # 1. Compute the discrepancy covariance and Q_d.
    K_d = squared_exponential_covariance(sensor_locations, sigma_d, l_d)  # (n_y, n_y)
    Q_d = K_d + sigma_e**2 * np.eye(n_y)
    
    # 2. Cholesky factorization of Q_d (with a small jitter for numerical stability)
    jitter = 1e-10
    L, lower = cho_factor(Q_d + jitter * np.eye(n_y), lower=True)
    
    # 3. Compute A = P^T @ Q_d^{-1} @ P.
    #    This is done by solving Q_d X = P for X, i.e., X = Q_d^{-1} P.
    invQd_P = cho_solve((L, lower), P)  # shape (n_y, n_u)
    A = P.T @ invQd_P                   # shape (n_u, n_u)
    
    # 4. Compute the posterior precision and covariance.
    #    C_{u|Y}^{-1} = C_u^{-1} + n_o * rho^2 * A.
    Cu_prior_inv = np.linalg.inv(Cu_prior + jitter * np.eye(Cu_prior.shape[0]))
    Posterior_precision = Cu_prior_inv + n_o * (rho**2) * A
    cov_u = np.linalg.inv(Posterior_precision)  # This is C_{u|Y}.
    
    # 5. Posterior mean update exactly as in the paper:
    #    u_{|Y} = C_{u|Y} * ( C_u^{-1} u_prior + rho * P^T Q_d^{-1} * (∑_{i=1}^{n_o} y_i) )
    sum_y = np.sum(Y, axis=1, keepdims=True) # sum of observations over i, shape (n_y,)
    invQd_sum_y = cho_solve((L, lower), sum_y)  # Q_d^{-1} * (∑_i y_i)
    u_post = cov_u @ (Cu_prior_inv @ u_prior + rho * (P.T @ invQd_sum_y).flatten())
    
    return u_post, cov_u


def squared_exponential_covariance(x, sigma, ls):
    """
    Example SE kernel, shape (len(x), len(x)).
    """
    x = np.atleast_2d(x).T
    diff = x - x.T
    return sigma**2 * np.exp(-0.5 * (diff / ls)**2)

# 1) Plot the 1D bar structure: FE nodes (blue) vs. sensor locations (red)
def plot_1d_structure(fe_nodes, sensor_locations):
    """
    Plot a 1D bar (black line from 0..1), FE nodes in blue, sensor locations in red.
    """
    plt.figure(figsize=(8, 2))

    # Draw the bar as a black line on x in [0,1], y=0
    plt.plot([0, 1], [0, 0], 'k-', linewidth=2, label='Structure')

    # FE nodes in blue
    plt.plot(fe_nodes, np.zeros_like(fe_nodes), 'bo', label='FE Nodes')

    # Sensors in red
    plt.plot(sensor_locations, np.zeros_like(sensor_locations), 'ro', label='Sensors')

    plt.ylim(-0.1, 0.1)  # small vertical range just to see the markers
    plt.xlabel("x in [0,1]")
    plt.title("1D Structure: FE Nodes (blue), Sensors (red), Bar (black line)")
    plt.legend()
    plt.show()

# 2) Plot histograms of MCMC chain for each hyperparameter
def plot_mcmc_histograms(chain, param_names, burn_in=0):
    """
    chain: (n_iter, d) array. d=3 for [rho, sigma_d, l_d]
    param_names: e.g. ["rho", "sigma_d", "l_d"]
    burn_in: how many initial samples to discard
    """
    chain_post = chain[burn_in:, :]
    inferred_means = np.mean(chain_post, axis=0)

    fig, axs = plt.subplots(1, len(param_names), figsize=(4*len(param_names), 4))
    for i, name in enumerate(param_names):
        ax = axs[i] if len(param_names) > 1 else axs
        ax.hist(chain_post[:, i], bins=50, density=True, alpha=0.5, color='C0')
        ax.axvline(x=inferred_means[i], color='r', linestyle='--', linewidth=2)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior of {name}")
    plt.tight_layout()
    plt.show()


# 3) Plot the FE prior solution +/- 95% along with the true system's latent data
def plot_fe_prior_and_true(fe_nodes, u_prior, Cu_prior, true_mean_func, true_cov_sigma, true_cov_ls):
    """
    fe_nodes, u_prior, Cu_prior: FE prior (size n_u).
    sensor_locations: length ny
    z_draws: shape (ny, no). The latent GP data from 'generate_true_response'.
             We'll plot the mean across columns (no).
    """
    std_prior = np.sqrt(np.diag(Cu_prior))  # (n_u,)
    lower_95_u = u_prior - 1.96*std_prior
    upper_95_u = u_prior + 1.96*std_prior

    # Mean of the true system:
    x = np.linspace(0,1,100)
    z_mean = true_mean_func(x) # shape (ny,)
    K_ts, _ = build_greens_covariance(x, true_cov_sigma, true_cov_ls, true_error = 0, nquad=30)
    std_ts = np.sqrt(np.diag(K_ts))
    lower_95_ts = z_mean - 1.96*std_ts
    upper_95_ts = z_mean + 1.96*std_ts

    plt.figure(figsize=(8, 5))
    # Plot prior mean and 95% band
    plt.plot(fe_nodes, u_prior, 'k-', label="FE Prior Mean")
    plt.fill_between(fe_nodes, lower_95_u, upper_95_u, color='grey', alpha=0.5,
                     label="95% Prior Interval")

    # Plot the "true" latent system only at sensor points
    plt.plot(x, z_mean, 'r-', label="True System Mean (sensors)")
    plt.fill_between(x, lower_95_ts, upper_95_ts, color='r', alpha=0.5,
                     label="95% Prior Interval")
    plt.xlabel("x in [0,1]")
    plt.ylabel("u or z")
    plt.title("FE Prior vs. True GP Latent")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Plot MCMC Chains for each hyperparameter ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    iterations = np.arange(mcmc_iterations)
    axs[0].plot(iterations, chain[:, 0])
    axs[0].set_ylabel('ρ')
    axs[0].set_title('MCMC Chain for ρ')
    
    axs[1].plot(iterations, chain[:, 1])
    axs[1].set_ylabel('σ_d')
    axs[1].set_title('MCMC Chain for σ_d')
    
    axs[2].plot(iterations, chain[:, 2])
    axs[2].set_ylabel('l_d')
    axs[2].set_xlabel('Iteration')
    axs[2].set_title('MCMC Chain for l_d')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_el = 32
    length_1d = 1.0
    dirich_nodes = [0, n_el]
    np.random.seed(42)

    # True hyperparameters
    # rho_true = 0.75
    # sigma_d_true = 0.025
    # l_d_true = 0.08

    def kappa_bar(x):
        return np.zeros_like(x) # np.log(0.7 + 0.3 * np.sin(2.0 * np.pi * x))

    sigma_k = 0.0
    ls_k = 0.25

    def f_bar(x):
        return (np.pi**2/5)*np.ones_like(x) # np.ones_like(x)

    sigma_f = 0.3
    ls_f = 0.25

    xpos_nodes, u_prior, Cu_prior = statFEM_1d(
        n_el, length_1d, dirich_nodes,
        kappa_bar, sigma_k, ls_k,
        f_bar, sigma_f, ls_f
    )


    # === Observation noise used in the likelihood model (sigma_e) ===
    sigma_e = np.sqrt(2.5*10**(-5))  # Standard deviation used in the inference likelihood
    
    # === Sensor and observation configuration ===
    ny = 33  # Number of sensors
    no = 1000  # Number of observation vectors (readings per sensor)
    domain = (0, 1)
    
    # --- Generate Data Using GP True System Response ---
    # Define the true system GP parameters (Eqs. (60–62)):
    # Here, we define the true mean as a function of x (sensor positions).
    true_mean_func = lambda x: 0.2 * np.sin(np.pi * x) + 0.02 * np.sin(7 * np.pi * x)
    true_cov_sigma = 0.15      # GP scaling factor (amplitude)
    true_cov_ls = 0.5         # GP lengthscale
    true_error = np.sqrt(2.5*10**(-5))         # Observation noise (added to the GP latent function)
    
    # Generate sensor locations (used both for the GP and for building the projection matrix in inference)
    sensor_locations = generate_sensor_locations(ny, domain)
    
    # Generate the latent true response z and observation matrix Y (ny x no)
    z, Y, K_z, K_y = generate_true_response(sensor_locations, true_mean_func, true_cov_sigma, true_cov_ls, true_error, no)
    print("Generated data using GP true system response.")

    # Generate data with 11 sensors and 100 readings
    # Y, sensor_locs, P = generate_synthetic_data_fe(
    #     u_prior, Cu_prior,
    #     rho_true, sigma_d_true, l_d_true,
    #     ny=11, no=100
    # )
    # print("Generated data using GP true FE response.")
    
    # === Set initial hyperparameters for MCMC inference (for [rho, sigma_d, l_d]) ===
    initial_hyperparams = [1, 0.1, 0.1]  # Starting guesses for [rho, sigma_d, l_d]
    mcmc_iterations = 20000
    
    # Optionally, define a log-prior function.
    def log_prior(w):
        rho, sigma_d, l_d = w
        if rho <= 0 or sigma_d <= 0 or l_d <= 0:
            return -np.inf
        return 0.0  # Uniform (improper) prior
    
    # Assume FE node positions are equally spaced in [0, 1]
    n_u = len(u_prior)
    fe_nodes = np.linspace(domain[0], domain[1], n_u)
    
    # Run inference using the FE prior and the GP-generated observation data Y.
    # (The FE prior provides u_prior and Cu_prior and, together with the projection matrix P,
    #  is used to evaluate the unconditional p(Y) and other conditionals.)
    chain, log_posteriors, sensor_locations, P = run_inference(
        u_prior, Cu_prior, Y, sigma_e, ny, no,
        fe_nodes=fe_nodes, domain=domain,
        initial_hyperparams=initial_hyperparams,
        mcmc_iterations=mcmc_iterations,
        proposal_cov=None,  # Let run_inference choose a default proposal covariance.
        prior_func=log_prior
    )
    
    # Postprocess the MCMC chain (e.g., discard burn-in, compute posterior means)
    burn_in = int(mcmc_iterations / 4)
    inferred_hyperparams = np.mean(chain[burn_in:], axis=0)

    # Compute 95% credible intervals (2.5% and 97.5% quantiles):
    lower_95ci = np.percentile(chain[burn_in:], 2.5, axis=0)
    upper_95ci = np.percentile(chain[burn_in:], 97.5, axis=0)

    # Print results for each hyperparameter:
    param_names = ["rho", "sigma_d", "l_d"]
    for i, name in enumerate(param_names):
        print(f"{name} mean = {inferred_hyperparams[i]:.4f}, "
            f"95% CI = [{lower_95ci[i]:.4f}, {upper_95ci[i]:.4f}]")
    
    # 1) Plot the 1D structure
    plot_1d_structure(fe_nodes, sensor_locations)

    # 2) Plot histograms of MCMC chain
    param_names = ["rho", "sigma_d", "l_d"]
    plot_mcmc_histograms(chain, param_names, burn_in=burn_in)

    # 3) Plot the FE prior solution vs. the "true system" latent data
    plot_fe_prior_and_true(fe_nodes, u_prior, Cu_prior, true_mean_func, true_cov_sigma, true_cov_ls)

    u_post, Cu_post = posterior_u_given_Y_iid(
    Y, P,
    u_prior, Cu_prior,
    inferred_hyperparams[0], inferred_hyperparams[1], inferred_hyperparams[2], sigma_e,
    sensor_locations
    )

    plot_u_prior_posterior_with_data(
    fe_nodes, u_prior, Cu_prior,
    u_post, Cu_post,
    sensor_locations, Y
    )
