import numpy as np
import matplotlib.pyplot as plt
from mesh import generate_mesh_1D, element_barycenters
from assembly import assemble_force_mean, build_local_stiffness, assemble_stiffness_and_partials
from boundary import build_reduced_system, expand_solution, expand_covariance
from covariance import build_force_cov, solve_covariance
from plotting import plot_forcing_term, plot_diffusion_coefficient

def statFEM_1d(n_el, len, dirich_nodes, kappa_bar, sigma_k, ell_k, f_bar, sigma_f, ell_f):
    # Mesh generation
    xpos_nodes = generate_mesh_1D(n_el, len)
    xpos_bary = element_barycenters(xpos_nodes)
    
    # Plot GPs
    plot_diffusion_coefficient(xpos_bary, kappa_bar, sigma_k, ell_k)
    plot_forcing_term(xpos_nodes, f_bar, sigma_f, ell_f, "Forcing Term (f(x))", "x (Node)", "f(x)")

    # Log-diffusivity covariance
    kappa_vals = kappa_bar(xpos_bary)
    X_bary = xpos_bary[:, None]
    dist2 = (X_bary - X_bary.T)**2
    Ck = sigma_k**2 * np.exp(-0.5 * dist2 / ell_k**2)

    # Assembly
    f_vec = assemble_force_mean(xpos_nodes, f_bar)
    K0_el = [build_local_stiffness(xpos_nodes[e], xpos_nodes[e+1]) for e in range(n_el)]
    A_bar, dA_list = assemble_stiffness_and_partials(kappa_vals, K0_el)

    # Boundary handling
    A_red, f_red, dA_red_list, Cf_red, free2g = build_reduced_system(
        A_bar, f_vec, dA_list, build_force_cov(xpos_nodes, sigma_f, ell_f), dirich_nodes
    )

    # Solve system
    u_red_bar = np.linalg.solve(A_red, f_red)
    Cu_red = solve_covariance(A_red, np.linalg.inv(A_red), Cf_red, Ck, f_red, dA_red_list)

    # Expand results
    u_bar_full = expand_solution(u_red_bar, free2g, n_el+1)
    Cu_full = expand_covariance(Cu_red, free2g, n_el+1)

    # Plot solution
    std_u = np.sqrt(np.diag(Cu_full))
    plt.figure(figsize=(7,5))
    plt.plot(xpos_nodes, u_bar_full, 'k-', label='Mean')
    plt.fill_between(xpos_nodes, u_bar_full-1.96*std_u, u_bar_full+1.96*std_u, color='grey', alpha=0.3)
    plt.xlabel("x"), plt.ylabel(r"$u_h(x)$"), plt.ylim(0, 0.25), plt.grid(True)
    plt.legend(), plt.savefig("SFEM_Sol.png", dpi=300), plt.close()

    return xpos_nodes, u_bar_full, Cu_full