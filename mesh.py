import numpy as np

def generate_mesh_1D(n_el, len):
    """
    Generate a 1D mesh with $n_el$ elements and total length $len$.

    Input:
    ----------
    n_el : int
        # elements in the mesh.
    len : float
        Length of the 1D domain.

    Output:
    -------
    xpos_nodes : ndarray, shape (n_nodes,)
        Array of nodal coordinates in 1D.
    """
    return np.linspace(0.0, len, n_el + 1)  # n_nodes = n_el + 1

def element_barycenters(xpos_nodes):
    """
    Compute the barycenter of each element for a 1D mesh (Midpoint between nodes if mass, or A e.g., is constant). For examples with varying A, SEE THIS.

    Input:
    ----------
    xpos_nodes : ndarray, shape (n_nodes,)
        Nodal coordinates of the 1D mesh.

    Output:
    -------
    xpos_bary : ndarray, shape (n_el,)
        Barycenter coordinates for each element.
    """
    xpos_n1 = xpos_nodes[:-1]
    xpos_n2 = xpos_nodes[1:]

    return 0.5 * (xpos_n1 + xpos_n2)  # n_el barycenters