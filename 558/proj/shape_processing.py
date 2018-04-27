import numpy as np
from scipy.spatial import Delaunay
import copy
import numpy.linalg as la
from sympy import *
from matplotlib import pyplot as plt
import pprint
import sys

EPS = 10 ** -8


# compares x and y with safety for precision
def compare(x, y):
    if abs(x - y) < EPS:
        return 0
    elif x - y > 0:
        return 1
    else:
        return -1


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


# computes whether v2 is on the left of v1
def on_left(v1, v2):
    return compare(np.cross(v1, v2), 0)


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


# finds where l1 intersects with the first of all possible lines, returns -1, -1 is it doesn't
def intersect(l1, posslines):
    v1 = np.subtract(l1[1], l1[0])
    for l2 in posslines:
        v2 = np.subtract(l2[1], l2[0])
        if abs(on_left(np.subtract(l1[0], l2[1]), v1) - on_left(np.subtract(l1[0], l2[0]), v1)) <= 1 or abs(
                on_left(np.subtract(l1[1], l2[1]), v2) - on_left(np.subtract(l1[0], l2[1]),
                                                                 v2)) <= 1:  # segments do not intersect
            continue
        else:  # find where lines, not segments, intersect
            if compare(d(l1[0], l2), 0) == 0 and compare(d(l1[1], l2), 0) == 0:  # segment started from this line (we ignore this)
                continue
            p1 = l1[0]
            p2 = l2[0]
            if v2[1] * v1[0] - v1[1] * v2[0] == 0:  # infinite solutions, ie same line (just return one of the points)
                return p2
            t = np.dot([-v1[1], v1[0]], [p1[0] - p2[0], p1[1] - p2[1]]) / (v2[1] * v1[0] - v1[1] * v2[0])
            val = p2[0] + t * v2[0], p2[1] + t * v2[1]
            return val
    return -1, -1


# returns whether returns the barycentric coords of p in Oedge
def get_bary(edge, p):
    matrix = [
        [edge[0][0], 0, edge[1][0]],
        [edge[0][1], 0, edge[1][1]],
        [1, 1, 1]
    ]
    if compare(la.det(matrix), 0) == 0:
        raise BaseException("Failed to invert matrix, have a triangle with no width")
    return np.matmul(la.inv(matrix), [p[0], p[1], 1])


def interp(square, x, y, h, f):
    bits = ''
    bits = bits.join([str(int((val > 0))) for val in square])
    for i in range(4):
        square[i] = abs(square[i])
    if bits == '1111' or bits == '0000':
        return [-1]
    elif bits == '1110' or bits == '0001':
        p1y = y - h * square[3] / (square[3] + square[0])
        p2x = x - h * square[2] / (square[3] + square[2])
        return [[x - h, p1y], [p2x, y]] if bits == '1110' else [[p2x, y], [x - h, p1y]]
    elif bits == '1101' or bits == '0010':
        p1y = y - h * square[2] / (square[1] + square[2])
        p2x = x - h * square[2] / (square[3] + square[2])
        return [[x, p1y], [p2x, y]] if bits == '0010' else [[p2x, y], [x, p1y]]
    elif bits == '1011' or bits == '0100':
        p1y = y - h * square[2] / (square[1] + square[2])
        p2x = x - h * square[1] / (square[0] + square[1])
        return [[x, p1y], [p2x, y - h]] if bits == '1011' else [[p2x, y - h], [x, p1y]]
    elif bits == '0111' or bits == '1000':
        p1y = y - h * square[3] / (square[3] + square[0])
        p2x = x - h * square[1] / (square[0] + square[1])
        return [[x - h, p1y], [p2x, y - h]] if bits == '1000' else [[p2x, y - h], [x - h, p1y]]
    elif bits == '1100' or bits == '0011':
        p1y = y - h * square[3] / (square[3] + square[0])
        p2y = y - h * square[2] / (square[1] + square[2])
        return [[x - h, p1y], [x, p2y]] if bits == '1100' else [[x, p2y], [x - h, p1y]]
    elif bits == '1001' or bits == '0110':
        p1x = x - h * square[1] / (square[1] + square[0])
        p2x = x - h * square[2] / (square[3] + square[2])
        return [[p1x, y - h], [p2x, y]] if bits == '1001' else [[p2x, y], [p1x, y - h]]
    elif bits == '1010' or bits == '0101':
        mid = f(x - h / 2, y - h / 2)
        if (mid <= 0 and bits == '1010') or (mid > 0 and bits == '0101'):
            p1x = x - h * square[1] / (square[1] + square[0])
            p2y = y - h * square[2] / (square[1] + square[2])
            p3y = y - h * square[3] / (square[0] + square[3])
            p4x = x - h * square[2] / (square[2] + square[3])
            return [[p1x, y - h], [x, p2y], [x - h, p3y], [p4x, y]] if bits == '1010' else [[p1x, y - h], [x, p2y],
                                                                                            [x - h, p3y], [p4x, y]]
        if (mid > 0 and bits == '1010') or (mid <= 0 and bits == '0101'):
            p1x = x - h * square[1] / (square[1] + square[0])
            p2y = y - h * square[3] / (square[0] + square[3])
            p3y = y - h * square[2] / (square[1] + square[2])
            p4x = x - h * square[2] / (square[2] + square[3])
            return [[p1x, y - h], [x - h, p2y], [x, p3y], [p4x, y]] if bits == '1010' else [[p1x, y - h], [x - h, p2y],
                                                                                            [x, p3y], [p4x, y]]


def march(func, xmin, xmax, ymin, ymax, h):
    x, y = symbols('x y')
    expr = sympify(func)
    f = lambdify((x, y), expr, "math")
    points1 = dict()
    points2 = dict()
    width = int(ceiling((xmax - xmin) / h) + 1)
    height = int(ceiling((ymax - ymin) / h) + 1)
    grid = np.zeros((width, height))
    p = []
    for i in range(width):
        for j in range(height):
            grid[i][j] = f(xmin + h * i, ymin + h * j)
            if i == 0 or j == 0:
                continue
            square = list()
            square[0:3] = [grid[i - 1][j - 1], grid[i][j - 1], grid[i][j], grid[i - 1][j]]
            news = interp(square, xmin + h * i, ymin + h * j, h, f)
            if len(news) == 1:
                continue
            elif len(news) == 2:
                print("Found a special point! The new guys are: " + str(news))
                if str(news[0]) in points1:
                    points2[str(news[0])] = news
                else:
                    points1[str(news[0])] = news
                if str(news[1]) in points1:
                    points2[str(news[1])] = news
                else:
                    points1[str(news[1])] = news
                p = news[0]
            elif len(news) == 4:
                for n in range(4):
                    if str(news[n]) in points1:
                        points2[str(news[n])] = news[2 * floor(n / 2):2 * floor(n / 2) + 2]
                    else:
                        points1[str(news[n])] = news[2 * floor(n / 2):2 * floor(n / 2) + 2]
                p = news[0]
    pprint.pprint(grid)
    points = list()
    edges = list()
    points.append(p)
    e = points1[str(p)]
    points1.pop(str(p))
    points2.pop(str(p))
    p = (e[1] if p == e[0] else e[0])
    i = 1
    while len(points1) != 0:
        points.append(p)
        prev = points[i - 1]
        edges.append([prev, p])
        e1 = points1[str(p)]
        points1.pop(str(p))
        e2 = points2[str(p)]
        points2.pop(str(p))
        e = (e2 if prev in e1 else e1)
        p = (e[1] if p == e[0] else e[0])
        i += 1
    edges.append([points[len(points) - 1], points[0]])
    return points, edges


# Tests whether p is inside the simply polygon defined by edges, returns 1 if in, 0 if on edge, and -1 if out
def in_poly(edges, p):
    sum = 0
    for edge in edges:
        orient = np.cross(np.subtract(edge[1], edge[0]), np.multiply(-1, edge[0]))
        sign = 1 if orient >= 0 else -1
        alphas = get_bary(edge, p)
        if all(compare(alpha, 0) >= 0 for alpha in alphas):  # On boundary
            count = 0
            for a in alphas:
                count = count + 1 if compare(a, 0) == 0 else count
            sum += .5 ** count * sign
        elif all(compare(alpha, 0) > 0 for alpha in alphas):  # Inside
            sum += sign
    if sum == .5:
        return 'on'
    return 'in' if sum >= 1 else 'out'


# Computes the distance between the boundary of that polygon and p
def dist(edges, p):
    min = sys.maxsize
    for edge in edges:
        dist = d(p, edge)
        if dist <= min:
            min = dist
    return min


# tests whether [a, b] is a diagonal in the polygon
def diagonal(i, j, points, edges):
    diag = np.subtract(points[i], points[j])
    lft = np.subtract(points[i-1], points[i])
    rt = np.subtract(points[i], points[i+1])
    if on_left(diag, lft) >= 0 and on_left(diag, rt) <= 0 and not intersect([points[i], points[j]], edges):
        return True
    return False


def initialize():
    


# Ear removal algorithm
def triangulate(edges, points):
    initialize()


def run(expression_file, points_file):
    with open(points_file, 'r') as in_file:
        file = in_file.read()
    points = [pair.split("\t") for pair in file.split("\n")]
    points.pop()
    for i in range(len(points)):
        points[i] = [float(points[i][0]), float(points[i][1])]
    with open(expression_file, 'r') as file:
        doc = file.read()
    lines = doc.splitlines()
    exp = lines[0]
    xmin, xmax, ymin, ymax = lines[1].split(', ')
    xmin, xmax, ymin, ymax = float(xmin), float(xmax), float(ymin), float(ymax)
    h = float(lines[2])
    fig, ax = plt.subplots()
    # Part a)
    poly_points, edges = march(exp, xmin, xmax, ymin, ymax, h)
    plot(poly_points, ax, 'bo')
    for line in edges:
        drawline(line[0], line[1], ax, 'b-')

    # Part b)
    for p in points:
        print('{0} : {1}, {2}'.format(p, in_poly(edges, p), dist(edges, p)))

    # Part c)

    plt.show()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
