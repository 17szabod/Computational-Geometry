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


def compute_center(face):
    p1, p2, p3 = face[0], face[1], face[2]
    m = np.array([
        [p2[0] - p1[0], p2[1] - p1[1]],
        [p2[0] - p3[0], p2[1] - p3[1]]
    ])
    return np.matmul(la.inv(m), [.5 * (p2[0] ** 2 - p1[0] ** 2 + p2[1] ** 2 - p1[1] ** 2),
                                 .5 * (p2[0] ** 2 - p3[0] ** 2 + p2[1] ** 2 - p3[1] ** 2)])


def voronoi_points(tri):
    points = tri.points
    face_names = tri.simplices
    vor_points = list()
    for name in face_names:
        face = points[name]
        center = compute_center(face)
        vor_points.append(center)
    return vor_points


def get_edges(tri):
    points = tri.points
    faces = points[tri.simplices]
    edges = np.array(list([[[0, 0], [0, 0]]]))
    for face in faces:
        e1 = [face[0], face[1]]
        if e1[1][0] < e1[0][0]:
            e1 = np.flipud(e1)
        e2 = [face[1], face[2]]
        if e2[1][0] < e2[0][0]:
            e2 = np.flipud(e2)
        e3 = [face[2], face[0]]
        if e3[1][0] < e3[0][0]:
            e3 = np.flipud(e3)
        edges = np.append(edges, [e1, e2, e3], axis=0)
    edges = np.delete(edges, 0, axis=0)
    _, idx = np.unique(edges, return_index=True, axis=0)
    edges = edges[np.sort(idx)]
    return edges


def crust(points):
    tri1 = Delaunay(points)
    vor_points = voronoi_points(tri1)
    edge1 = get_edges(tri1)
    points.extend(vor_points)
    tri2 = Delaunay(points)
    edge2 = get_edges(tri2)
    copy1 = list(copy.deepcopy(edge1))
    copy2 = list(copy.deepcopy(edge2))
    [copy1.__setitem__(i, str(copy1[i])) for i in range(len(copy1))]
    [copy2.__setitem__(i, str(copy2[i])) for i in range(len(copy2))]
    result = edge1[np.in1d(copy1, copy2)]
    return result


def run(filename):
    with open(filename, 'r') as in_file:
        file = in_file.read()
    points = [pair.split("\t") for pair in file.split("\n")]
    for i in range(len(points)):
        points[i] = [float(points[i][0]), float(points[i][1])]
    points.pop()
    fig, ax = plt.subplots()
    plot(points, ax, 'b.')
    result = crust(points)
    for edge in result:
        drawline(edge[0], edge[1], ax, 'r-')
    plt.show()
    return


if __name__ == "__main__":
    run(sys.argv[1])
