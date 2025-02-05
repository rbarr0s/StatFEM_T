import numpy as np

def lumped_integral_phi(xpos_nodes):
    """
    Compute lumped integrals for nodal shape functions phi_i(x).

    Input:
    ----------
    xpos_nodes : ndarray, shape (n_nodes,)
        Nodal coordinates of the 1D mesh.

    Output:
    -------
    lumps : ndarray, shape (n_nodes,)
        Lumped integrals for each node.
    """
    n_nodes = len(xpos_nodes)
    lumps = np.zeros(n_nodes)
    # boundary lumps
    lumps[0]          = 0.5*(xpos_nodes[1] - xpos_nodes[0])
    lumps[n_nodes-1]  = 0.5*(xpos_nodes[n_nodes-1] - xpos_nodes[n_nodes-2])
    # internal lumps
    for i in range(1, n_nodes-1):
        lumps[i] = 0.5*(xpos_nodes[i]   - xpos_nodes[i-1]) \
                 + 0.5*(xpos_nodes[i+1] - xpos_nodes[i])
    return lumps

def build_force_cov(xpos_nodes, sigma_f, ell_f):
    """
    Build the lumped approximation for the forcing covariance matrix (based on the mass matrices defined by the Ref. paper):
      (C_f)_{ij} approx (∫ phi_i) * cov_kernel(x_i, x_j) * (∫ phi_j)

    Input:
    ----------
    xpos_nodes : ndarray, shape (n_nodes,)
        Array of nodal coordinates in 1D.
    sigma_f, ell_f : problem input

    Output:
    -------
    Cf : ndarray, shape (n_nodes, n_nodes)
        Lumped forcing covariance matrix.
    """
    lumps = lumped_integral_phi(xpos_nodes)
    X = xpos_nodes[:, None]
    dist2 = (X - X.T)**2
    Cf = np.outer(lumps, lumps) * sigma_f**2 * np.exp(-0.5*dist2/ell_f**2)
    return Cf

def solve_covariance(A_red, Ainv_red, Cf_red, Ck, f_red, dA_red_list):
    """
    Compute the solution covariance matrix C_u in the reduced space.

    Input:
    ----------
    A_red : ndarray, shape (n_free, n_free)
        Reduced stiffness matrix.
    Ainv_red : ndarray, shape (n_free, n_free)
        Precomputed inverse of the reduced stiffness matrix.
    Cf_red : ndarray, shape (n_free, n_free)
        Reduced covariance matrix of the forcing term.
    Ck : ndarray, shape (n_el, n_el)
        Covariance matrix of the random field (log-diffusivity).
    f_red : ndarray, shape (n_free,)
        Reduced mean force vector.
    dA_red_list : list of ndarray, each of shape (n_free, n_free)
        Reduced partial derivatives of the stiffness matrix w.r.t. kappa.

    Output:
    -------
    Cu_red : ndarray, shape (n_free, n_free)
        Reduced covariance matrix of the solution.
    """
    # Contribution from forcing covariance
    Cu_red = Ainv_red @ Cf_red @ Ainv_red.T # 1st term, from original response w/o approximation

    # Contribution from stochastic kappa
    Cf_plus_ffT = Cf_red + np.outer(f_red, f_red)  # Cf + f*f^T
    n_el = Ck.shape[0]
    for e in range(n_el):
        for d in range(n_el):
            coeff = Ck[e, d]  # (C_kappa)_{e,d}
            if abs(coeff) < 1e-14:
                continue

            # Derivatives of A with respect to kappa_e and kappa_d
            dAl = dA_red_list[e]
            dAd = dA_red_list[d]

            # Correct order of operations for the second term
            term = (
                coeff
                * Ainv_red @ dAl @ Ainv_red @ Cf_plus_ffT @ Ainv_red.T @ dAd.T @ Ainv_red.T
            )

            Cu_red += term # 2nd term, comes from approximation and lenghty development
    return Cu_red