import matplotlib.pyplot as plt
import numpy as np

def _plot_gp(x, mean, lower, upper, title, ylabel, ylim, fname):
    plt.figure(figsize=(7,5))
    plt.plot(x, mean, 'k-', label='Mean')
    plt.fill_between(x, lower, upper, color='grey', alpha=0.3, label='95% CI')
    plt.xlabel("x"), plt.ylabel(ylabel), plt.ylim(*ylim)
    plt.title(title), plt.grid(True), plt.legend()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_forcing_term(x, mean_fn, sigma, ell, label, xlabel, ylabel):
    mean = mean_fn(x)
    X = x[:, None]
    cov = sigma**2 * np.exp(-0.5*(X - X.T)**2/ell**2)
    std = np.sqrt(np.diag(cov))
    _plot_gp(x, mean, mean-1.96*std, mean+1.96*std, 
            f"GP for {label}", ylabel, (0,1.5), "SFEM_Forc.png")

def plot_diffusion_coefficient(x, kappa_mean_fn, sigma_k, ell_k):
    kappa_mean = kappa_mean_fn(x)
    X = x[:, None]
    cov = sigma_k**2 * np.exp(-0.5*(X - X.T)**2/ell_k**2)
    mu_mean = np.exp(kappa_mean)
    std_mu = np.sqrt(np.diag(cov)) * mu_mean  # Approximation
    _plot_gp(x, mu_mean, mu_mean-1.96*std_mu, mu_mean+1.96*std_mu,
            "GP for $\\mu(x)$", "$\\mu(x)$", (0,1.5), "SFEM_DCoeff.png")