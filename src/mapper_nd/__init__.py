"""mapper-nd: the Mapper algorithm (Topological Data Analysis) from scratch.

The distinguishing feature is a filter to an arbitrary-dimensional space R^n, so the
cover uses overlapping hypercubes in n dimensions and the nerve is a full simplicial
complex (edges, triangles, tetrahedra, ...) rather than only a graph (1-skeleton).
"""

__version__ = "0.1.0"
