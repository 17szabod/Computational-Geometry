import voronoi
import numpy as np
import copy
import numpy.linalg as la
from matplotlib import pyplot as plt
import sys


def shape(tri, vor_points):
    points = tri.points
    face_names = tri.simplices
    shape_edges = np.array(list([[[0, 0], [0, 0]]]), ndmin=2)
    for name in range(len(face_names)):
        neighbors = tri.neighbors[name]
        orig = copy.deepcopy(neighbors)
        face = points[tri.simplices[name]]
        if np.any(neighbors[:] == -1):
            to_delete = list()
            for i in range(3):
                # Always choose the Delaunay edge if on the boundary
                if neighbors[i] == -1:
                    line = np.delete(face, i, axis=0)
                    shape_edges = np.append(shape_edges, [line], axis=0)
                    to_delete.append(i)
            neighbors = np.delete(neighbors, to_delete)
        for neighbor in neighbors:
            index = np.where(orig == neighbor)
            tmp = copy.deepcopy(face)
            tmp = np.delete(tmp, index, axis=0)  # Delaunay edge
            if should_flip(tmp[0], tmp[1], vor_points[neighbor], vor_points[name]):
                tmp = np.array([vor_points[neighbor], vor_points[name]])
            if tmp[1][0] < tmp[0][0]:
                tmp = np.flipud(tmp)
            shape_edges = np.append(shape_edges, [tmp], axis=0)
    shape_edges = np.delete(shape_edges, 0, axis=0)
    _, idx = np.unique(shape_edges, return_index=True, axis=0)
    shape_edges = shape_edges[np.sort(idx)]
    return shape_edges


def should_flip(a, b, q, p):
    a_dot = np.dot(np.subtract(a, q), np.subtract(a, p)) / (la.norm(np.subtract(a, q)) * la.norm(np.subtract(a, p)))
    b_dot = np.dot(np.subtract(b, q), np.subtract(b, p)) / (la.norm(np.subtract(b, q)) * la.norm(np.subtract(b, p)))
    if a_dot + b_dot > 0:  # and triangle.intersect([a, b], [q, p]):
        return True
    else:
        return False


def run(filename):
    with open(filename, 'r') as in_file:
        file = in_file.read()
    points = [pair.split("\t") for pair in file.split("\n")]
    for i in range(len(points)):
        points[i] = [float(points[i][0]), float(points[i][1])]
    points.pop()
    fig, ax = plt.subplots()
    tri = voronoi.Delaunay(points)
    points = tri.points
    v_points, v_edges = voronoi.voronoi(tri)
    edges = shape(tri, v_points)
    voronoi.plot(v_points, ax, 'y.')
    voronoi.plot(points, ax, 'b.')
    for edge in edges:
        voronoi.drawline(edge[0], edge[1], ax, 'r-')
    plt.show()


if __name__ == '__main__':
    run(sys.argv[1])
