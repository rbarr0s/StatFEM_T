import numpy as np

gauss_xi_1d = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
gauss_w_1d  = np.array([1.0, 1.0])

def shape_func_linear(xi):
    """
    Returns [phi1, phi2] for 1D linear shape funcs on [-1,1].
      phi1(xi) = (1 - xi)/2
      phi2(xi) = (1 + xi)/2
    """
    return np.array([
        0.5*(1.0 - xi),
        0.5*(1.0 + xi)
    ])


def assemble_force_mean(xpos_nodes, f_bar):
    """
    Assemble the consistent forcing vector:
      f[i] = ∫ f_bar(x)*phi_i(x) dx, integrated by Gaussian Quadrature (2 pts for full integration, element by element)

    Input:
    ----------
    xpos_nodes : ndarray, shape (n_nodes,)
        Nodal coordinates of the 1D mesh.
    f_bar : problem input
        Function for the mean forcing term, f_mean(x).

    Output:
    -------
    F : ndarray, shape (n_nodes,)
        Global forcing vector.
    """
    n_el = len(xpos_nodes) - 1
    n_nodes = len(xpos_nodes)
    F = np.zeros(n_nodes)
    
    for e in range(n_el):
        xL = xpos_nodes[e]
        xR = xpos_nodes[e+1]
        Le = xR - xL
        Fe = np.zeros(2)  # local (element) 2x1 vector
        for k in range(len(gauss_xi_1d)):
            xi = gauss_xi_1d[k]
            w  = gauss_w_1d[k]
            # parametric mapping
            x_phys = xL + 0.5*(xi+1.0)*Le
            J      = 0.5*Le
            phi    = shape_func_linear(xi)  # [phi1, phi2]
            
            f_val  = f_bar(x_phys)
            
            Fe[0] += w * f_val * phi[0] * J
            Fe[1] += w * f_val * phi[1] * J
        
        # assemble Fe into global vector
        iL, iR = e, e+1
        F[iL]   += Fe[0]
        F[iR]   += Fe[1]
    
    return F

def build_local_stiffness(xL, xR):
    """
    Build local stiffness matrix for linear elements for constant A. (If want to consider varying A, again SEE THIS)

    Input:
    ----------
    xR, xL : float
        Nodal coordinates of x1, x2 in each element

    Output:
    -------
    K0_e : ndarray, shape (2 x 2)
        Local stiffness matrix.
    """
    Le = xR - xL
    return (1./Le)*np.array([[1, -1],
                             [-1, 1]])

def assemble_stiffness_and_partials(kappa_el, K0_el):
    """
    Assemble the global stiffness matrix K for 1D Poisson problem:
      K_ij = ∫ exp(kappa(x)) * dphi_i/dx * dphi_j/dx dx

    Input:
    ----------
    K0_el : ndarray, shape (n_nodes, n_nodes)
        Base local stiffness matrix for each element.
    kappa_el : ndarray, shape (n_el,)
        Diffusion coefficient values at element barycenters.

    Output:
    -------
    K (A as general system matrix) : ndarray, shape (n_nodes, n_nodes)
        Global stiffness matrix.
    dK_dkappa : list of ndarray, shape (n_nodes, n_nodes) per element
        Derivatives of K w.r.t. each element's kappa.
    """
    n_el = len(kappa_el)
    n_nodes = n_el + 1
    A = np.zeros((n_nodes,n_nodes))
    dA = [np.zeros((n_nodes,n_nodes)) for _ in range(n_el)]
    for e in range(n_el):
        mu_e = np.exp(kappa_el[e])
        K0   = K0_el[e]
        iL, iR = e, e+1
        
        # Add local:
        A[iL,iL] += mu_e*K0[0,0]
        A[iL,iR] += mu_e*K0[0,1]
        A[iR,iL] += mu_e*K0[1,0]
        A[iR,iR] += mu_e*K0[1,1]
        
        # Partial wrt kappa_e:
        dA[e][iL,iL] += mu_e*K0[0,0]
        dA[e][iL,iR] += mu_e*K0[0,1]
        dA[e][iR,iL] += mu_e*K0[1,0]
        dA[e][iR,iR] += mu_e*K0[1,1]
    
    return A, dA
