def basis_to_triangle_map(V: 'np.ndarray[float]', T: 'np.ndarray[int]', E: list):
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
    for v in range(len(V)):
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

def local2global(V: 'np.ndarray[float]', T: 'np.ndarray[int]', E: list) -> dict:
    '''
    Given a set of matrices, a connectivity matrix T, and a set of edges E,
    computes a local 2 global map that for each triangle maps a local node
    index to a global node index - twelve per triangle.
    :param V: vertices
    :param T: connectivity matrix
    :param E: edges
    :return : dictionary -> triangle -> [global indices]
    '''

    l2g = {}
    edge_idx = {}

    N = len(V)

    # assign a global index to each edge
    for i, e in enumerate(E):
        edge_idx[e] = 3*N + i

    # assign a global index to each vertex, corresponding to evaluation and gradient
    for i in range(len(T)):
        v1, v2, v3 = T[i]
        e1, e2, e3 = get_one_edge(T[i])
        e1, e2, e3 = edge_idx[e1], edge_idx[e2], edge_idx[e3]
        l2g[i] = [v1, v1 + N, v1 + 2*N, e1,\
                  v2, v2 + N, v2 + 2*N, e2,\
                  v3, v3 + N, v3 + 2*N, e3]

    return l2g


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

def get_one_edge(t: 'np.ndarray[int]') -> list:
    '''
    Given a triangle t = [i1, i2, i3], returns the list of edges (i1, i2), (i2,
    i3), (i3, i1) where each tuple is sorted by index.
    :param t: single triangle
    :return: edges of triangle
    '''
    edges = [tuple(sorted([t[i], t[(i+1) % 3]])) for i in range(3)]
    return edges

def get_all_edges(T: 'np.ndarray[int]') -> list:
    '''
    Given a connectivity matrix T, returns a list of all the edges in the
    triangulation.
    :param T: connectivity matrix
    :return: list of all edges
    '''
    E = []
    for t in T:
        E += get_one_edge(t)
    return list(set(E))
