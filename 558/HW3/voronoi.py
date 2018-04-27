import numpy as np
from scipy.spatial import Delaunay
import copy
import numpy.linalg as la
from matplotlib import pyplot as plt
import sys


# plots every point in points on ax
def plot(points, ax, color):
    [ax.plot(p[0], p[1], color) for p in points]
    return


# draws a line with the given color (doesn't show on its own)
def drawline(p1, p2, ax, color):
    v = np.subtract(p2, p1)
    alpha = np.linspace(0, 1)
    pointsx = np.add(p1[0], np.multiply(alpha, v[0]))
    pointsy = np.add(p1[1], np.multiply(alpha, v[1]))
    ax.plot(pointsx, pointsy, color)  # plots them in a matlab-esque way
    return  # turns out this is pretty stupid, really could just draw a line with 2 points


# finds where l1 intersects with the first of all possible lines, returns -1, -1 is it doesn't
def intersect(p, l1, *posslines):
    v1 = np.subtract(l1[1], l1[0])
    vals = list()
    for l2 in posslines:
        v2 = np.subtract(l2[1], l2[0])
        p1 = l1[0]
        p2 = l2[0]
        if v2[1] * v1[0] - v1[1] * v2[0] == 0:  # infinite solutions, ie same line (just return one of the points)
            return p2
        t = np.dot([-v1[1], v1[0]], [p1[0] - p2[0], p1[1] - p2[1]]) / (v2[1] * v1[0] - v1[1] * v2[0])
        val = [p2[0] + t * v2[0], p2[1] + t * v2[1]]
        vals.append(val)
    if len(vals) == 0:
        return -1, -1
    else:
        return min(vals, key=lambda x: la.norm(np.subtract(x, p)))


def compute_bounding_rectangle(points):
    x_max = max(points, key=lambda x: x[0])[0]
    x_min = min(points, key=lambda x: x[0])[0]
    y_max = max(points, key=lambda x: x[1])[1]
    y_min = min(points, key=lambda x: x[1])[1]
    const = (x_max - x_min + y_max - y_min) / 16
    x_max += const
    x_min -= const
    y_max += const
    y_min -= const
    # rec = [top_left, top_right, bottom_right, bottom_left]
    rec = [[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]
    plt.axis([x_min, x_max, y_min, y_max])
    return rec


def compute_center(face):
    p1, p2, p3 = face[0], face[1], face[2]
    m = np.array([
        [p2[0] - p1[0], p2[1] - p1[1]],
        [p2[0] - p3[0], p2[1] - p3[1]]
    ])
    return np.matmul(la.inv(m), [.5 * (p2[0] ** 2 - p1[0] ** 2 + p2[1] ** 2 - p1[1] ** 2),
                                 .5 * (p2[0] ** 2 - p3[0] ** 2 + p2[1] ** 2 - p3[1] ** 2)])


def voronoi(tri):
    points = tri.points
    rect = compute_bounding_rectangle(points)
    top = [rect[0], rect[1]]
    right = [rect[1], rect[2]]
    bottom = [rect[2], rect[3]]
    left = [rect[3], rect[0]]
    face_names = tri.simplices
    vor_points = list()
    vor_edges = np.array(list([[[0, 0], [0, 0]]]), ndmin=3)
    for name in face_names:
        face = points[name]
        center = compute_center(face)
        vor_points.append(center)
    # voronoi points have been computed
    for name in range(len(face_names)):
        neighbors = tri.neighbors[name]
        if np.any(neighbors[:] == -1):
            face = points[tri.simplices[name]]
            to_delete = list()
            for i in range(len(neighbors)):
                if neighbors[i] == -1:
                    line = np.delete(face, i, axis=0)
                    line = [np.divide(np.add(line[0], line[1]), 2), vor_points[name]]
                    vor_edges = np.append(vor_edges,
                                          [np.array([intersect(vor_points[name], line, top, left, bottom, right),
                                                     vor_points[name]])],
                                          axis=0)
                    to_delete.append(i)
            neighbors = np.delete(neighbors, to_delete)
        for neighbor in neighbors:
            tmp = copy.deepcopy(np.array([vor_points[neighbor], vor_points[name]]))
            if tmp[1][0] < tmp[0][0]:
                tmp = np.flipud(tmp)
            vor_edges = np.append(vor_edges, [tmp], axis=0)
    vor_edges = np.delete(vor_edges, 0, axis=0)
    _, idx = np.unique(vor_edges, return_index=True, axis=0)
    vor_edges = vor_edges[np.sort(idx)]
    return vor_points, vor_edges


def run(filename):
    with open(filename, 'r') as in_file:
        file = in_file.read()
    points = [pair.split("\t") for pair in file.split("\n")]
    for i in range(len(points)):
        points[i] = [float(points[i][0]), float(points[i][1])]
    points.pop()
    fig, ax = plt.subplots()
    tri = Delaunay(points)
    vor_points, edges = voronoi(tri)
    plot(vor_points, ax, 'y.')
    plot(points, ax, 'b.')
    for edge in edges:
        drawline(edge[0], edge[1], ax, 'r-')
    plt.show()
    return


if __name__ == "__main__":
    run(sys.argv[1])
