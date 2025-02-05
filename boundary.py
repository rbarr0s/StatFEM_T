import numpy as np

def build_reduced_system(A, f, dA_list, Cf, dirich_nodes):
    """
    Build the reduced system by eliminating Dirichlet boundary conditions (Direct stiffness method)

    Input:
    ----------
    A : ndarray, shape (n_nodes, n_nodes)
        Global stiffness matrix.
    f : ndarray, shape (n_nodes,)
        Global force vector.
    dA_list : list of ndarray, each of shape (n_nodes, n_nodes)
        Partial derivatives of A w.r.t. kappa for each element.
    Cf : ndarray, shape (n_nodes, n_nodes)
        Global forcing covariance matrix.
    dirich_nodes : list of int
        Indices of nodes where Dirichlet boundary conditions are applied.

    Output:
    -------
    A_red : ndarray, shape (n_free, n_free)
        Reduced stiffness matrix.
    f_red : ndarray, shape (n_free,)
        Reduced force vector.
    dA_red_list : list of ndarray, each of shape (n_free, n_free)
        Reduced partial derivatives of A.
    Cf_red : ndarray, shape (n_free, n_free)
        Reduced forcing covariance matrix.
    free2g : ndarray, shape (n_free,)
        Mapping of free degrees of freedom to global indices.
    """
    n_nodes = A.shape[0]
    fixed = set(dirich_nodes) # BC nodes
    all_dofs = set(range(n_nodes)) # BC DoFs (Not really useful in 1D bar element, but SEE THIS for other types)
    free_dofs = sorted(list(all_dofs - fixed)) 
    free2g = np.array(free_dofs, dtype=int) # Mapping of free DoFs
    
    # reduce A, f, Direct Stiffness Method
    A_red = A[free2g[:,None], free2g] # Reduced stiffness matrix (n_free x n_free)
    f_red = f[free2g] # Reduced force vector (n_free,)
    
    # reduce dA
    dA_red_list = []
    for dA in dA_list:
        tmp = dA[free2g[:,None], free2g]
        dA_red_list.append(tmp)
    
    # reduce Cf
    Cf_red = Cf[free2g[:,None], free2g] # Reduced covariance (n_free x n_free)
    
    return A_red, f_red, dA_red_list, Cf_red, free2g

def expand_solution(u_red, free2g, n_nodes):
    """
    Expand the reduced solution to the full solution vector

    Input:
    ----------
    u_red : ndarray, shape (n_free,)
        Reduced solution vector.
    free2g : ndarray, shape (n_free,)
        Mapping of free degrees of freedom to global indices.
    n_nodes : int
        Total number of nodes.

    Output:
    -------
    u_full : ndarray, shape (n_nodes,)
        Full solution vector.
    """
    u_full = np.zeros(n_nodes)
    u_full[free2g] = u_red
    return u_full

def expand_covariance(Cu_red, free2g, n_nodes):
    """
    Expand the reduced covariance matrix to the full system

    Input:
    ----------
    Cu_red : ndarray, shape (n_free, n_free)
        Reduced covariance matrix.
    free2g : ndarray, shape (n_free,)
        Mapping of free degrees of freedom to global indices.
    n_nodes : int
        Total number of nodes.

    Output:
    -------
    Cu_full : ndarray, shape (n_nodes, n_nodes)
        Full covariance matrix.
    """
    Cu_full = np.zeros((n_nodes,n_nodes))
    for i_red,i_g in enumerate(free2g):
        for j_red,j_g in enumerate(free2g):
            Cu_full[i_g,j_g] = Cu_red[i_red,j_red]
    return Cu_full