def construct_basis_mapping(V: 'np.ndarray[float]', T: 'np.ndarray[int]', E: list):
    """
    Given a connectivity matrix T, a set of vertices V, and a set of edges E,
    returns a dictionary mapping global basis number to the set of triangles it
    is defined on.
    :param V: set of vertices
    :param T: connectivity matrix
    :return: dictionary mapping basis number to set of triangles.
    """
    basis_mapping = {}
    basis_counter = 0
    # value and gradient interpolation
    for v in range(V):
        I = incident_triangles(v, T)
        basis_mapping[basis_counter] = I
        basis_mapping[basis_counter + len(V)] = I
        basis_mapping[basis_counter + 2*len(V)] = I
        basis_counter += 1

    # edge_normal interpolation
    basis_counter = len(basis_mapping)
    for e in E:
        I = adjacent_triangles(e, T)
        basis_mapping[basis_counter] = I
        basis_counter += 1
    return basis_mapping


def incident_triangles(v: int, T: 'np.ndarray[int]') -> list:
    '''
    Given the connectivity matrix T and a vertex v,
    return a list of the triangles that contain vertex v.
    :param v: vertex number
    :param T: connectivity matrix T
    :return: list of the triangles
    '''

    triangle_idx = []
    for i in range(len(T)):
        if v in T[i]:
            triangle_idx.append(i)
    return triangle_idx

def adjacent_triangles(e: tuple, T: 'np.ndarray[int]') -> list:
    '''
    Given the connectivity matrix T and an edge e,
    return a list of the triangles that contain edge e.
    :param e: edge
    :param T: connectivity matrix T
    :return: list of the 1 or 2 triangles
    '''

    triangle_idx = []
    for i in range(len(T)):
        if e[0] in T[i] and e[1] in T[i]:
            triangle_idx.append(i)
    return triangle_idx

