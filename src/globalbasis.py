import numpy as np
from helper_functions import basis_to_triangle_map, local2global, get_all_edges
from SSplines.src.splinespace import SplineSpace
from SSplines.src.triangle_functions import sample_triangle_uniform

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.spatial as sp

class GlobalBasisSpline(object):

    def __init__(self, triangles_with_support):
        self.T = triangles_with_support
        pass

class CompositeSpline(object):
    def __init__(self, coefficients, basis, local_to_global_mapping):
        self.c = coefficients
        self.b = basis
        self.l2g = local_to_global_mapping

    def __call__(self, x, k):
        s = 0
        for i in self.l2g[k]:
            s += self.c[i] * self.b[i](x, k)
        return s

    def lapl(self, x, k):
        s = 0
        for i in self.l2g[k]:
            s += self.c[i] * self.b[i].lapl(x, k)
        return s

class GlobalSplineSpace(object):

    def __init__(self, V, T):
        '''

        '''
        self.V = V
        self.T = T
        self.int_edges, self.bnd_edges = get_all_edges(self.T)
        self.E = self.int_edges + self.bnd_edges
        self.dim = 3*len(V) + len(self.E)
        self.basis_to_triangle_map = basis_to_triangle_map(V, self.T, self.E)
        self.local_to_global_map, self.edge_idx = local2global(V, self.T, self.E)
        self.global_to_local_map = self.global2local()

        self.local_spline_spaces = [SplineSpace(V[t], 2) for t in T] # precompute the spline spaces
        self.global_basis = [self._construct_global_basis_function(i) for i in range(self.dim)]
        self.boundary_nodes = self._get_bnd_nodes()
        self.interior_nodes = [i for i in range(self.dim) if i not in self.boundary_nodes]


    def _get_bnd_nodes(self):
        '''
        Returns the indices of the global basis functions that lie on the boundary.
        '''
        bnd_nodes = []

        bnd_verts = []
        for E in self.bnd_edges:
            bnd_verts += list(E)
        bnd_verts = list(set(bnd_verts))

        N = len(self.V)
        for v in bnd_verts:
            bnd_nodes += [v, v+N, v+2*N]
        for e in self.bnd_edges:
            bnd_nodes.append(self.edge_idx[e])

        return bnd_nodes

    def _construct_global_basis_function(self, i: int) -> CompositeSpline:
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
        if i >= 3*len(self.V) and len(triangles_with_support) == 2:
            local_basis_repr[triangle_id] = B*-1

        return self.GlobalBasis(local_basis_repr, triangles_with_support)

    class GlobalBasis(object):
        def __init__(self, local_basis_repr, triangles_with_support):
            self.local_basis_repr = local_basis_repr
            self.triangles_with_support = triangles_with_support

        def __call__(self, x, k):
            # if x lies in a supported triangle, evaluate
            if k in self.triangles_with_support:
                return self.local_basis_repr[k](x)
            # else, return 0
            else:
                return np.zeros(len(x))

        def lapl(self, x, k):
            # if x lies in a supported triangle, evaluate
            if k in self.triangles_with_support:
                return self.local_basis_repr[k].lapl(x)
            # else, return 0
            else:
                return np.zeros(len(x))

        def grad(self, x, k):
            # if x lies in a supported triangle, evaluate
            if k in self.triangles_with_support:
                return self.local_basis_repr[k].grad(x)
            # else, return 0
            else:
                return np.zeros(len(x))

        def div(self, x, k):
            # if x lies in a supported triangle, evaluate
            if k in self.triangles_with_support:
                return self.local_basis_repr[k].div(x)
            # else, return 0
            else:
                return np.zeros_like(len(x))

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

    def function(self, coefficients):

        return CompositeSpline(coefficients, self.global_basis, self.local_to_global_map)


if __name__ == "__main__":
    V = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.5, 0.5]
    ])
    T = sp.Delaunay(V, incremental=True).simplices
    G = GlobalSplineSpace(V, T)
    points = [sample_triangle_uniform(V[T[i]], 30) for i in range(len(T))]
    tris = [sp.Delaunay(p) for p in points]

    for i, b in enumerate(G.global_basis):
        fig = plt.figure()
        axs = Axes3D(fig)
        axs.set_zlim(-1, 1)
        for t in range(len(T)):
            z = b(points[t], t)
            axs.plot_trisurf(points[t][:, 0], points[t][:, 1], z, triangles=tris[t].simplices)
        plt.title('$\\varphi_{%d}$' % i)
        plt.show()
    fig = plt.figure()
    axs = Axes3D(fig)
    c = np.array([1 + 0.025*i for i in range(G.dim)])
    f = G.function(c)
    for t in range(len(T)):
        z = f(points[t], t)
        axs.plot_trisurf(points[t][:, 0], points[t][:, 1], z, triangles=tris[t].simplices)

    plt.show()

