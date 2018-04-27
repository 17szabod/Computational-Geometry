import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
import pprint
import bisect
import timeit
import copy
import numpy.linalg as la
import functools
from matplotlib import pyplot as plt
import sys


# computes and sets EPS and EPS2
def computeEPS(points):
    variance = np.var(points)
    # variance = varx**2 + vary**2
    xmin = min(points)
    xmax = max(points)
    global EPS, EPS2
    EPS = variance ** .5 * (xmax[0] - xmin[0]) * pow(10, -6) / len(points) ** 2
    EPS2 = variance ** .5 * (xmax[0] - xmin[0]) * pow(10, -6) / len(points) ** 2


# compares x and y with safety for precision
def compare(x, y):
    if abs(x - y) < EPS:
        return 0
    elif x - y < 0:
        return 1
    else:
        return -1


def norm(v):
    sum = 0
    for x in v:
        sum += x ** 2
    return sum


# plots every point in points on ax
def plot(points, ax):
    [ax.plot(p[0], p[1], 'go') for p in points]
    return


# checks if a and b have the same sign
def same_sign(a, b):
    if compare(a, 0) == 0 or compare(b, 0) == 0:
        return False
    return compare(a / abs(a), b / abs(b)) == 0


# computes if the lines intersect
def intersect(l1, l2):
    v1 = np.subtract(l1[1], l1[0])
    v2 = np.subtract(l2[1], l2[0])
    return not (same_sign(np.cross(np.subtract(l1[0], l2[1]), v1),
                          np.cross(np.subtract(l1[0], l2[0]), v1)) or same_sign(
        np.cross(np.subtract(l1[1], l2[1]), v2), np.cross(np.subtract(l1[0], l2[1]), v2)))  # segments do not intersect


# distance between a point and line segment
def d(p, line):
    v = np.subtract(line[1], line[0])
    if 0 <= np.dot(v, np.subtract(p, line[0])) / (v[0] ** 2 + v[1] ** 2) <= 1:
        a = la.det([
            [line[0][0], line[0][1], 1],
            [line[1][0], line[1][1], 1],
            [p[0], p[1], 1]
        ])
        return abs(a / la.norm(np.subtract(line[1], line[0])))
    else:
        d_p0 = la.norm(np.subtract(line[0], p))
        d_p1 = la.norm(np.subtract(line[1], p))
        return min(d_p1, d_p0)


# returns whether point p is in face
def is_in(face, p):
    matrix = [
        [face[0][0], face[1][0], face[2][0]],
        [face[0][1], face[1][1], face[2][1]],
        [1, 1, 1]
    ]
    if compare(la.det(matrix), 0) == 0:
        return True
    alphas = np.matmul(la.inv(matrix), [p[0], p[1], 1])
    if all(a >= 0 for a in alphas):
        return True
    else:
        return False


# returns whether point p is on the boundary of face
def is_on(face, p):
    matrix = [
        [face[0][0], face[1][0], face[2][0]],
        [face[0][1], face[1][1], face[2][1]],
        [1, 1, 1]
    ]
    alphas = np.matmul(la.inv(matrix), [p[0], p[1], 1])
    if any(abs(a - 0) < EPS2 for a in alphas):
        return True
    else:
        return False


# draws a line with the given color (doesn't show on its own)
def drawline(p1, p2, ax, color):
    v = np.subtract(p2, p1)
    alpha = np.linspace(0, 1)
    pointsx = np.add(p1[0], np.multiply(alpha, v[0]))
    pointsy = np.add(p1[1], np.multiply(alpha, v[1]))
    ax.plot(pointsx, pointsy, color)  # plots them in a matlab-esque way
    return  # turns out this is pretty stupid, really could just draw a line with 2 points


# Graham Scan Algorithm for convex hull, wrote it myself from last assignment
def convex_hull(points, ax):
    p0 = min(points)
    points.remove(p0)

    def cmp(x, y):
        val = compare(np.cross(np.subtract(y, p0), np.subtract(x, p0)), 0)
        if val == 0:
            return compare(la.norm(np.subtract(x, p0)), la.norm(np.subtract(y, p0)))
        return val

    points.sort(key=functools.cmp_to_key(cmp))
    stack = list()  # a stack to hold all the points on the hull, ideal data structure
    stack.append(p0)
    for point in points:
        while compare(np.cross(np.subtract(stack[len(stack) - 1], stack[len(stack) - 2]),
                               np.subtract(point, stack[len(stack) - 1])), 0) < 0:
            stack.pop()
        stack.append(point)
    while compare(np.cross(np.subtract(stack[len(stack) - 1], stack[len(stack) - 2]),
                           np.subtract(p0, stack[len(stack) - 1])), 0) < 0:
        stack.pop()
    for i in range(len(stack) - 1):
        drawline(stack[i], stack[i + 1], ax, '--r')
    drawline(stack[len(stack) - 1], p0, ax, '--r')
    points.append(p0)
    return stack


# Recursively walks through the triangulation to find p
def walk(edges, p, curr):
    if len(curr[2]) != 0 and is_in(curr[2], p):
        return curr, True
    if is_in(curr[3], p):
        return curr, False
    side = np.cross(np.subtract(curr[1], curr[0]), np.subtract(p, curr[0]))
    if side <= 0:  # on right
        f = curr[3]
        u1 = edges[curr[6]]
        u2 = edges[curr[7]]
    else:  # on left
        f = curr[2]
        u1 = edges[curr[5]]
        u2 = edges[curr[4]]
    l1 = [f[1], curr[0]]
    l2 = [f[1], curr[1]]
    vals1 = [d(p, l1), d(p, l2)]
    vals2 = [
        abs(la.det([
            [l1[0][0], l1[0][1], 1],
            [l1[1][0], l1[1][1], 1],
            [p[0], p[1], 1],
        ]) / la.norm(np.subtract(l1[1], l1[0]))),
        abs(la.det([
            [l2[0][0], l2[0][1], 1],
            [l2[1][0], l2[1][1], 1],
            [p[0], p[1], 1]
        ]) / la.norm(np.subtract(l2[1], l2[0])))
    ]
    if vals1[1] < vals1[0]:
        return walk(edges, p, u2)
    elif vals1[1] > vals1[0]:
        return walk(edges, p, u1)
    else:
        if vals2[1] < vals2[0]:
            return walk(edges, p, u1)
        else:
            return walk(edges, p, u2)


# Returns edge, onLeft
def find_me(edges, mid, p):
    return walk(edges, p, edges["ch" + str(mid)])


def should_flip(edge, edges):
    if len(edge[2]) == 0:
        return False
    a, b, c, d = edge[3][1], edge[2][1], edge[0], edge[1]
    a_dot = np.dot(np.subtract(a, c), np.subtract(a, d)) / (la.norm(np.subtract(a, c)) * la.norm(np.subtract(a, d)))
    b_dot = np.dot(np.subtract(b, c), np.subtract(b, d)) / (la.norm(np.subtract(b, c)) * la.norm(np.subtract(b, d)))
    c_dot = np.dot(np.subtract(c, a), np.subtract(c, b)) / (la.norm(np.subtract(c, a)) * la.norm(np.subtract(c, b)))
    d_dot = np.dot(np.subtract(d, a), np.subtract(d, b)) / (la.norm(np.subtract(d, a)) * la.norm(np.subtract(d, b)))
    if a_dot + b_dot < c_dot + d_dot and intersect([a, b], [c, d]):
        return True
    elif a_dot + b_dot > c_dot + d_dot or not intersect([a, b], [c, d]):
        return False
    else:
        pprint.pprint(edges)
        print(edge)
        raise BaseException("Error, cocircular points: points " + str([a, b, c, d]) + " are cocircular")


def flip(e, edges, faces, vertices):
    a, b, c, d = e[3][1], e[2][1], e[0], e[1]
    e1 = edges[e[4]]
    e2 = edges[e[5]]
    e3 = edges[e[6]]
    e4 = edges[e[7]]
    tmp = copy.deepcopy(e[2])
    tmp.sort()
    if faces.__contains__(str(tmp)):
        faces.pop(str(tmp))
    tmp = copy.deepcopy(e[3])
    tmp.sort()
    if faces.__contains__(str(tmp)):
        faces.pop(str(tmp))
    e = [a, b, [a, c, b], [a, d, b], e2[8], e3[8], e4[8], e1[8], e[8]]
    edges[e[8]] = e
    if e1[0] == b:
        e1[3], e1[6], e1[7] = [e1[0], a, e1[1]], e[8], e4[8]
    else:
        e1[2], e1[4], e1[5] = [e1[0], a, e1[1]], e[8], e4[8]
    if e2[1] == b:
        e2[3], e2[6], e2[7] = [e2[0], a, e2[1]], e3[8], e[8]
    else:
        e2[2], e2[4], e2[5] = [e2[0], a, e2[1]], e3[8], e[8]
    if e3[0] == a:
        e3[3], e3[6], e3[7] = [e3[0], b, e3[1]], e[8], e2[8]
    else:
        e3[2], e3[4], e3[5] = [e3[0], b, e3[1]], e[8], e2[8]
    if e4[1] == a:
        e4[3], e4[6], e4[7] = [e4[0], b, e4[1]], e1[8], e[8]
    else:
        e4[2], e4[4], e4[5] = [e4[0], b, e4[1]], e1[8], e[8]
    tmp = [a, c, b]
    tmp.sort()
    faces[str(tmp)] = [[a, c, b], e[8]]
    tmp = [a, d, b]
    tmp.sort()
    faces[str(tmp)] = [[a, d, b], e[8]]
    if vertices[str(c)][1] == e[8]:
        vertices[str(c)][1] = e2[8]
    if vertices[str(d)][1] == e[8]:
        vertices[str(d)][1] = e1[8]
    return edges, faces, vertices


# triangulates the points and prints the output, returning face,
def triangulate(points):
    fig, ax = plt.subplots()
    edges = dict()
    faces = dict()
    vertices = dict()
    ch = convex_hull(copy.deepcopy(points), ax)
    for p in ch:
        points.remove(p)
    for i in range(len(ch)):
        if i == 0:
            edges["ch" + str(i)] = [ch[i], ch[(i + 1) % len(ch)], [], [ch[i], ch[i + 2], ch[i + 1]], "", "", "edge0",
                                    "ch1", "ch" + str(i)]
        elif i == 1:
            edges["ch" + str(i)] = [ch[i], ch[(i + 1) % len(ch)], [], [ch[1], ch[0], ch[2]], "", "", "ch0", "edge0",
                                    "ch" + str(i)]
        elif i == len(ch) - 2:
            edges["ch" + str(i)] = [ch[i], ch[(i + 1) % len(ch)], [], [ch[i], ch[0], ch[i + 1]], "", "",
                                    "edge" + str(i - 2), "ch" + str(i + 1), "ch" + str(i)]
        elif i == len(ch) - 1:
            edges["ch" + str(i)] = [ch[i], ch[(i + 1) % len(ch)], [], [ch[i], ch[i - 1], ch[0]], "", "",
                                    "ch" + str(i - 1), "edge" + str(i - 3), "ch" + str(i)]
        else:
            edges["ch" + str(i)] = [ch[i], ch[(i + 1) % len(ch)], [], [ch[i], ch[0], ch[(i + 1) % len(ch)]], "", "",
                                    "edge" + str(i - 2), "edge" + str(i - 1), "ch" + str(i)]
        vertices[str(ch[i])] = [ch[i], "ch" + str(i)]
    p0 = ch[0]
    for i in range(len(ch) - 3):
        edge = [ch[i + 2], p0, [p0, ch[i + 3], ch[i + 2]], [p0, ch[i + 1], ch[i + 2]], "edge" + str(i + 1),
                "ch" + str(i + 2), "ch" + str(i + 1), "edge" + str(i - 1), "edge" + str(i)]
        if i == 0:
            edge[7] = "ch0"
        if i == len(ch) - 4:
            edge[4] = "ch" + str(i + 3)
            tmp = [p0, ch[i + 3], ch[i + 2]]
            tmp.sort()
            faces[str(tmp)] = [tmp, "edge" + str(i)]
        edges["edge" + str(i)] = edge
        tmp = [p0, ch[i + 1], ch[i + 2]]
        tmp.sort()
        faces[str(tmp)] = [tmp, "edge" + str(i)]
    for i in range(len(points)):
        p = points[i]
        edge, on_left = find_me(edges, int(np.floor(len(ch) / 2)), p)
        if is_on(edge[2] if on_left else edge[3], p):
            if on_left:
                other = edges[edge[4]] if d(p, edges[edge[4]][0:2]) < d(p, edges[edge[5]][0:2]) else edges[edge[5]]
            else:
                other = edges[edge[6]] if d(p, edges[edge[6]][0:2]) < d(p, edges[edge[7]][0:2]) else edges[edge[7]]
            edge = edge if d(p, edge[0:2]) < d(p, other[0:2]) else other
            fl = edge[2]
            if len(fl) == 0:
                continue
            fr = edge[3]
            tmp = [fr[1], edge[0], p]
            tmp.sort()
            faces[str(tmp)] = [tmp, "a" + str(i)]
            tmp = [fr[1], edge[1], p]
            tmp.sort()
            faces[str(tmp)] = [tmp, "a" + str(i)]
            tmp = [fl[1], edge[0], p]
            tmp.sort()
            faces[str(tmp)] = [tmp, "b" + str(i)]
            tmp = [fl[1], edge[1], p]
            tmp.sort()
            faces[str(tmp)] = [tmp, "b" + str(i)]
            edges["n1sp" + str(i)] = [p, edge[1], [p, fl[1], edge[1]], [p, fr[1], edge[1]], edge[4], "b" + str(i),
                                      "a" + str(i), edge[7], "n1sp" + str(i)]
            edges["n2sp" + str(i)] = [edge[0], p, [p, fl[1], edge[0]], [p, fr[1], edge[0]], "b" + str(i), edge[5],
                                      edge[6], "a" + str(i), "n2sp" + str(i)]
            edges["b" + str(i)] = [fl[1], p, [fl[1], edge[1], p], [fl[1], edge[0], p], "n1sp" + str(i), edge[4],
                                   edge[5], "n2sp" + str(i), "b" + str(i)]
            edges["a" + str(i)] = [fr[1], p, [fr[1], edge[0], p], [fr[1], edge[1], p], "n2sp" + str(i), edge[6],
                                   edge[7], "n1sp" + str(i), "a" + str(i)]
            vertices[str(p)] = [p, "n1sp" + str(i)]
            # fix other edges
            e = edges[edge[4]]
            if e[0] == edge[1]:
                edges[edge[4]] = [e[0], e[1], [e[0], p, e[1]], e[3], "b" + str(i), "n1sp" + str(i), e[6], e[7], e[8]]
            else:
                edges[edge[4]] = [e[0], e[1], e[2], [e[0], p, e[1]], e[4], e[5], "b" + str(i), "n1sp" + str(i), e[8]]
            e = edges[edge[5]]
            if e[1] == edge[0]:
                edges[edge[5]] = [e[0], e[1], [e[0], p, e[1]], e[3], "n2sp" + str(i), "b" + str(i), e[6], e[7], e[8]]
            else:
                edges[edge[5]] = [e[0], e[1], e[2], [e[0], p, e[1]], e[4], e[5], "n2sp" + str(i), "b" + str(i), e[8]]
            e = edges[edge[6]]
            if e[0] == edge[0]:
                edges[edge[6]] = [e[0], e[1], [e[0], p, e[1]], e[3], "a" + str(i), "n2sp" + str(i), e[6], e[7], e[8]]
            else:
                edges[edge[6]] = [e[0], e[1], e[2], [e[0], p, e[1]], e[4], e[5], "a" + str(i), "n2sp" + str(i), e[8]]
            e = edges[edge[7]]
            if e[1] == edge[1]:
                edges[edge[7]] = [e[0], e[1], [e[0], p, e[1]], e[3], "n1sp" + str(i), "a" + str(i), e[6], e[7], e[8]]
            else:
                edges[edge[7]] = [e[0], e[1], e[2], [e[0], p, e[1]], e[4], e[5], "n1sp" + str(i), "a" + str(i), e[8]]
            edges.pop(edge[8])
            tmp = [fl[0], fl[1], fl[2]]
            tmp.sort()
            faces.pop(str(tmp))
            tmp = [fr[0], fr[1], fr[2]]
            tmp.sort()
            faces.pop(str(tmp))
            continue
        if on_left:
            face = copy.deepcopy(edge[2])
            edges["na" + str(i)] = [edge[0], p, [edge[0], face[1], p], [edge[0], edge[1], p], "nc" + str(i),
                                    edge[5], edge[8], "nb" + str(i), "na" + str(i)]
            edges["nb" + str(i)] = [edge[1], p, [edge[1], edge[0], p], [edge[1], face[1], p], "na" + str(i),
                                    edge[8], edge[4], "nc" + str(i), "nb" + str(i)]
            edges["nc" + str(i)] = [face[1], p, [face[1], edge[1], p], [face[1], edge[0], p], "nb" + str(i),
                                    edge[4], edge[5], "na" + str(i), "nc" + str(i)]
            edge1 = edges[edge[5]]
            edge2 = edges[edge[4]]
            if edge1[1] == edge[0]:
                edge1[2] = [edge1[0], p, edge1[1]]
                edge1[4], edge1[5] = "na" + str(i), "nc" + str(i)
            else:
                edge1[3] = [edge1[0], p, edge1[1]]
                edge1[6], edge1[7] = "na" + str(i), "nc" + str(i)
            if edge2[0] == edge[1]:
                edge2[2] = [edge2[0], p, edge2[1]]
                edge2[4], edge2[5] = "nc" + str(i), "nb" + str(i)
            else:
                edge2[3] = [edge2[0], p, edge2[1]]
                edge2[6], edge2[7] = "nc" + str(i), "nb" + str(i)
            edge[2] = [edge[0], p, edge[1]]
            edge[4], edge[5] = "nb" + str(i), "na" + str(i)
        else:
            face = copy.deepcopy(edge[3])
            edges["na" + str(i)] = [edge[0], p, [edge[0], edge[1], p], [edge[0], face[1], p], "nb" + str(i),
                                    edge[8], edge[6], "nc" + str(i), "na" + str(i)]
            edges["nb" + str(i)] = [edge[1], p, [edge[1], face[1], p], [edge[1], edge[0], p], "nc" + str(i),
                                    edge[7], "na" + str(i), edge[8], "nb" + str(i)]
            edges["nc" + str(i)] = [face[1], p, [face[1], edge[0], p], [face[1], edge[1], p], "na" + str(i),
                                    edge[6], edge[7], "nb" + str(i), "nc" + str(i)]
            edge1 = edges[edge[6]]
            edge2 = edges[edge[7]]
            if edge1[0] == edge[0]:
                edge1[2] = [edge1[0], p, edge1[1]]
                edge1[4], edge1[5] = "nc" + str(i), "na" + str(i)
            else:
                edge1[3] = [edge1[0], p, edge1[1]]
                edge1[6], edge1[7] = "nc" + str(i), "na" + str(i)
            if edge2[1] == edge[1]:
                edge2[2] = [edge2[0], p, edge2[1]]
                edge2[4], edge2[5] = "nb" + str(i), "nc" + str(i)
            else:
                edge2[3] = [edge2[0], p, edge2[1]]
                edge2[6], edge2[7] = "nb" + str(i), "nc" + str(i)
            edge[3] = [edge[0], p, edge[1]]
            edge[6], edge[7] = "na" + str(i), "nb" + str(i)
        vertices[str(p)] = [p, "na" + str(i)]
        tmp = [edge[0], edge[1], p]
        tmp.sort()
        faces[str(tmp)] = [tmp, "na" + str(i)]
        tmp = [edge[1], face[1], p]
        tmp.sort()
        faces[str(tmp)] = [tmp, "nc" + str(i)]
        tmp = [face[1], edge[0], p]
        tmp.sort()
        faces[str(tmp)] = [tmp, "nc" + str(i)]
        tmp = copy.deepcopy(face)
        tmp.sort()
        faces.pop(str(tmp))
    return edges, faces, vertices


def badness(e, edges):
    edge = edges[e]
    if len(edge[2]) == 0:
        return sys.maxsize
    a, b, c, d = edge[3][1], edge[2][1], edge[0], edge[1]
    theta1 = np.cross(np.subtract(a, c), np.subtract(a, d)) / (la.norm(np.subtract(a, c)) * la.norm(np.subtract(a, d)))
    if np.dot(np.subtract(a, c), np.subtract(a, d)) < 0:
        theta1 += 1
    theta2 = np.cross(np.subtract(b, c), np.subtract(b, d)) / (la.norm(np.subtract(b, c)) * la.norm(np.subtract(b, d)))
    if np.dot(np.subtract(b, c), np.subtract(b, d)) < 0:
        theta2 += 1
    phi1 = np.cross(np.subtract(c, a), np.subtract(c, b)) / (la.norm(np.subtract(c, a)) * la.norm(np.subtract(c, b)))
    if np.dot(np.subtract(c, a), np.subtract(c, b)) < 0:
        phi1 += 1
    phi2 = np.cross(np.subtract(d, a), np.subtract(d, b)) / (la.norm(np.subtract(d, a)) * la.norm(np.subtract(d, b)))
    if np.dot(np.subtract(d, a), np.subtract(d, b)) < 0:
        phi2 += 1
    return (phi1 + phi2) / (theta1 + theta2)


def flip_edges(edges, faces, vertices):
    fig, ax = plt.subplots()
    queue = list(edges.keys())
    for name in queue:
        if name.startswith("ch"):
            queue.remove(name)
        elif not intersect([edges[name][3][1], edges[name][2][1]], [edges[name][0], edges[name][1]]):
            queue.remove(name)
    have_flipped = dict()
    for point in vertices:
        have_flipped[point] = list()
    queue.sort(key=lambda x: badness(x, edges))
    while len(queue) != 0:
        edge = queue.pop(0)
        e = edges[edge]
        if should_flip(e, edges):
            edges, faces, vertices = flip(e, edges, faces, vertices)
            loc = have_flipped[str(e[0])]
            loc.append(e[1])
            for i in range(4):
                next_edge = edges[e[i + 4]]
                if should_flip(next_edge, edges) and (
                        not have_flipped[str(next_edge[0])].__contains__(next_edge[1]) or not have_flipped[
                    str(next_edge[1])].__contains__(next_edge[0])):
                    queue.append(e[i + 4])
    # plot everything
    for vertex in vertices:
        vertex = vertices[vertex]
        ax.plot(vertex[0][0], vertex[0][1], 'ro')
    for edge in edges:
        edge = edges[edge]
        drawline(edge[0], edge[1], ax, 'b')


def run(filename):
    with open(filename, 'r') as in_file:
        file = in_file.read()
    points = [pair.split("\t") for pair in file.split("\n")]
    for i in range(len(points)):
        points[i] = [float(points[i][0]), float(points[i][1])]
    points.pop()
    computeEPS(points)
    fig, ax = plt.subplots()
    tri = Delaunay(points)
    delaunay_plot_2d(tri, ax=ax)
    plt.show()
    return
    edges, faces, vertices = triangulate(points)
    flip_edges(edges, faces, vertices)
    print("number of triangles = " + str(len(faces)))
    print("number of edges = " + str(len(edges)))
    print("number of vertices = " + str(len(vertices)))
    plt.show()
    return


if __name__ == "__main__":
    run(sys.argv[1])
