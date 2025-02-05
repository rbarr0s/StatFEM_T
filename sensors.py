import numpy as np

def generate_sensor_locations(ny, domain=(0, 1)):
    """
    Generate equally spaced sensor locations over the given domain.
    
    :param ny: Number of sensors.
    :param domain: Tuple (min, max) for the domain.
    :return: NumPy array of sensor locations (length ny).
    """
    return np.linspace(domain[0], domain[1], ny)

def build_projection_matrix(sensor_locations, fe_nodes):
    """
    Build the projection matrix P that maps the FE solution u (defined at fe_nodes)
    to sensor locations using linear interpolation.
    
    :param sensor_locations: NumPy array of sensor positions (ny,).
    :param fe_nodes: NumPy array of finite element node positions (n_u,).
    :return: Projection matrix P of shape (ny, n_u).
    """
    ny = len(sensor_locations)
    n_u = len(fe_nodes)
    P = np.zeros((ny, n_u))
    
    for i, s in enumerate(sensor_locations):
        # For sensor s, find indices for linear interpolation.
        if s <= fe_nodes[0]:
            P[i, 0] = 1.0
        elif s >= fe_nodes[-1]:
            P[i, -1] = 1.0
        else:
            # Find j such that fe_nodes[j] <= s < fe_nodes[j+1]
            j = np.searchsorted(fe_nodes, s) - 1
            x0 = fe_nodes[j]
            x1 = fe_nodes[j+1]
            # Linear interpolation weights:
            w1 = (s - x0) / (x1 - x0)
            w0 = 1 - w1
            P[i, j] = w0
            P[i, j+1] = w1
    return P
