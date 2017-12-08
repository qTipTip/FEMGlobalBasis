import numpy as np
from helper_functions import basis_to_triangle_map, local2global, get_all_edges
from SSplines.src.splinespace import SplineSpace
from SSplines.src.triangle_functions import sample_triangle_uniform

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.spatial as sp

class GlobalBasisFunction(object):

    def __init__(self, V, T, c):
        '''

        '''
        self.V = V
        self.T = T
        self.C = C


class GlobalSplineSpace(object):

    def __init__(self, V, T):
        '''

        '''
        self.V = V
        self.T = T
        self.E = get_all_edges(self.T)
        self.dim = 3*len(V) + len(self.E)

        self.basis_to_triangle_map = basis_to_triangle_map(V, self.T, self.E)
        self.local_to_global_map = local2global(V, self.T, self.E)
        self.global_to_local_map = self.global2local()

        self.local_spline_spaces = [SplineSpace(V[t], 2) for t in T] # precompute the spline spaces
        self.global_basis = [self._construct_global_basis_function(i) for i in range(self.dim)]

    def _construct_global_basis_function(self, i: int) -> GlobalBasisFunction:
        '''
        Constructs a callable composite spline function corresponding to global
        basis number i.
        '''
        triangles_with_support = self.basis_to_triangle_map[i]
        local_basis_repr = {}
        for triangle_id in triangles_with_support:
            S = self.local_spline_spaces[triangle_id]
            B = S.basis(type='H')[self.global_to_local_map[i][triangle_id]]
            local_basis_repr[triangle_id] = B


        # if the global basis function is defined over an internal edge, then
        # flip the sign of one of the local basis functions to preserve continuity
        if i >= 3*len(V) and len(triangles_with_support) == 2:
            local_basis_repr[triangle_id] = B*-1

        def globalbasis(x, k):
            # if x lies in a supported triangle, evaluate
            if k in triangles_with_support:
                return local_basis_repr[k](x)
            # else, return 0
            else:
                return np.zeros(len(x))

        return globalbasis

    def global2local(self):
        '''
        Constructs a map that for each global basis function
        assigns a dictionary mapping triangle of support to the local representation
        of the global basis function on that triangle.
        '''
        B = {}
        N = len(self.V)

        for vertex_id in range(N):
            I = self.basis_to_triangle_map[vertex_id]
            B[vertex_id] = {j : self.local_to_global_map[j].index(vertex_id) for j in I}
            B[vertex_id + N] = {j : self.local_to_global_map[j].index(vertex_id + N) for j in I}
            B[vertex_id + 2*N] = {j : self.local_to_global_map[j].index(vertex_id + 2*N) for j in I}

        for edge_id in range(3*N, 3*N + len(self.E)):
            I = self.basis_to_triangle_map[edge_id]
            B[edge_id] = {j : self.local_to_global_map[j].index(edge_id) for j in I}

        return B
if __name__ == "__main__":
    V = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    T = sp.Delaunay(V).simplices

    G = GlobalSplineSpace(V, T)
    points = [sample_triangle_uniform(V[T[i]], 25) for i in range(len(T))]
    tris = [sp.Delaunay(p) for p in points]
    for i, b in enumerate(G.global_basis):
        fig = plt.figure()
        axs = Axes3D(fig)
        axs.set_zlim3d(0, 1)
        for t in range(len(T)):
            z = b(points[t], t)
            axs.plot_trisurf(points[t][:, 0], points[t][:, 1], z, triangles=tris[t].simplices)
        plt.title('$\\varphi_{%d}$' % i)
        plt.show()

