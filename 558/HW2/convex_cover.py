import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys
import functools

EPS = pow(10, -8)


# compares x and y with safety for precision
def compare(x, y):
    if abs(x - y) < EPS:
        return 0
    elif x - y < 0:
        return 1
    else:
        return -1


# plots every point in points on ax
def plot(points, ax):
    [ax.plot(p[0], p[1], 'go') for p in points]
    return


# unused, would compute the angle between v1 and v2
def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (la.norm(v1) * la.norm(v2)))


# draws a line with the given color (doesn't show on its own)
def drawline(p1, p2, ax, color):
    v = np.subtract(p2, p1)
    alpha = np.linspace(0, 1)
    pointsx = np.add(p1[0], np.multiply(alpha, v[0]))
    pointsy = np.add(p1[1], np.multiply(alpha, v[1]))
    ax.plot(pointsx, pointsy, color)  # plots them in a matlab-esque way
    return  # turns out this is pretty stupid, really could just draw a line with 2 points


# Graham Scan Algorithm for convex hull, wrote it myself
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
    return stack


# deprecated, was implementing a different algorithm-- oops
def get_antipodal(i, ch):
    n = len(ch)
    p = ch[i]
    v1 = np.subtract(ch[i], ch[(i - 1) % n])
    v2 = np.subtract(ch[(i + 1) % n], ch[i])
    maxi = p
    for point in ch:
        if np.cross(np.subtract(point, p), v1) > np.cross(np.subtract(maxi, p), v1):
            maxi = point
        elif np.cross(np.subtract(point, p), v1) < np.cross(np.subtract(maxi, p), v1):
            break
    q = maxi
    maxi = ch[(i + 1) % n]
    for point in ch:
        if np.cross(np.subtract(point, p), v2) > np.cross(np.subtract(maxi, p), v2):
            maxi = point
        elif np.cross(np.subtract(point, p), v2) < np.cross(np.subtract(maxi, p), v2):
            break
    s = maxi
    return range(ch.index(q), ch.index(s))


# returns the unit vector of p2 - p1
def get_unit(p2, p1):
    v = np.subtract(p2, p1)
    return np.divide(v, la.norm(v))


# computes the area of the rectangle corresponding to the current state of inds and lines
def compute_rec(inds, lines, ch):
    p = ch[inds[0]]
    q = np.add(ch[inds[0]], lines[0])
    width = np.cross(np.subtract(ch[inds[2]], p), np.subtract(ch[inds[2]], q))
    p = ch[inds[1]]
    q = np.add(ch[inds[1]], lines[1])
    length = np.cross(np.subtract(ch[inds[3]], p), np.subtract(ch[inds[3]], q))
    return abs(length * width)


# computes where the lines defined by p and v intersect
def intersect(p1, v1, p2, v2):
    if v2[1] * v1[0] - v1[1] * v2[0] == 0:  # infinite solutions, ie same line (just return one of the points)
        return p2
    t = np.dot([-v1[1], v1[0]], [p1[0] - p2[0], p1[1] - p2[1]]) / (v2[1] * v1[0] - v1[1] * v2[0])
    val = p2[0] + t * v2[0], p2[1] + t * v2[1]
    return val


# finds the area of the convex hull and minimum bounding rectangle and draws them
def convex_cover(points):
    fig, ax = plt.subplots()
    plot(points, ax)
    ch = convex_hull(points, ax)  # compute the convex hull
    n = len(ch)
    inds = [
        ch.index(min(ch)),
        ch.index(max(ch, key=lambda z: z[1])),
        ch.index(max(ch)),
        ch.index(min(ch, key=lambda z: z[1]))
    ]
    min_area = abs((ch[inds[2]][0] - ch[inds[0]][0]) * (ch[inds[1]][1] - ch[inds[3]][1])) + 1
    # min_area just has to have at least 1 rectangle smaller than it, which is the first rectangle calculated
    min_rec = tuple()
    charea = 0
    for i in range(n):
        charea += ch[i][0] * ch[(i + 1) % n][1] - ch[i][1] * ch[(i + 1) % n][0]
    charea = abs(charea) / 2
    lines = [
        [0, -1], [-1, 0], [0, 1], [1, 0]
    ]
    for count in range(n):
        area = compute_rec(inds, lines, ch)
        if area < min_area:
            min_area = area
            min_rec = (inds, lines)
        thetas = [  # costly in terms of actual efficiency, but not asymptotically as it is still O(n)
            abs(np.cross(get_unit(ch[(inds[0] - 1) % n], ch[inds[0]]), lines[0])) if np.dot(
                get_unit(ch[(inds[0] - 1) % n], ch[inds[0]]), lines[0]) > 0 else abs(  # greater than 90
                np.cross(get_unit(ch[(inds[0] - 1) % n], ch[inds[0]]), lines[0])) + 1,
            abs(np.cross(get_unit(ch[(inds[1] - 1) % n], ch[inds[1]]), lines[1])) if np.dot(
                get_unit(ch[(inds[1] - 1) % n], ch[inds[1]]), lines[1]) > 0 else abs(  # greater than 90
                np.cross(get_unit(ch[(inds[1] - 1) % n], ch[inds[1]]), lines[1])) + 1,
            abs(np.cross(get_unit(ch[(inds[2] - 1) % n], ch[inds[2]]), lines[2])) if np.dot(
                get_unit(ch[(inds[2] - 1) % n], ch[inds[2]]), lines[2]) > 0 else abs(  # greater than 90
                np.cross(get_unit(ch[(inds[2] - 1) % n], ch[inds[2]]), lines[2])) + 1,
            abs(np.cross(get_unit(ch[(inds[3] - 1) % n], ch[inds[3]]), lines[3])) if np.dot(
                get_unit(ch[(inds[3] - 1) % n], ch[inds[3]]), lines[3]) > 0 else abs(  # greater than 90
                np.cross(get_unit(ch[(inds[3] - 1) % n], ch[inds[3]]), lines[3])) + 1
        ]
        i = thetas.index(min(thetas))
        lines[i] = get_unit(ch[(inds[i] - 1) % n], ch[inds[i]])  # next unit vector to use
        inds[i] = (inds[i] - 1) % n  # next index to use
        for x in range(3):  # just modify each to 90 degree rotations of the ones before it
            lines[(i + x + 1) % 4] = [lines[(i + x) % 4][1], -lines[(i + x) % 4][0]]
    # Draw the rectangle
    corners = [
        intersect(ch[min_rec[0][0]], min_rec[1][0], ch[min_rec[0][3]], min_rec[1][3]),
        intersect(ch[min_rec[0][0]], min_rec[1][0], ch[min_rec[0][1]], min_rec[1][1]),
        intersect(ch[min_rec[0][2]], min_rec[1][2], ch[min_rec[0][1]], min_rec[1][1]),
        intersect(ch[min_rec[0][2]], min_rec[1][2], ch[min_rec[0][3]], min_rec[1][3])
    ]
    drawline(corners[0], corners[1], ax, '--b')
    drawline(corners[1], corners[2], ax, '--b')
    drawline(corners[2], corners[3], ax, '--b')
    drawline(corners[3], corners[0], ax, '--b')
    return min_area, charea


def run(filename):
    with open(filename, 'r') as in_file:
        file = in_file.read()
    points = [pair.split("\t") for pair in file.split("\n")]
    # The input_file_generator makes an \n at the end, which I usually delete manually
    for i in range(len(points)):
        points[i] = [float(points[i][0]), float(points[i][1])]
    rec_area, ch_area = convex_cover(points)
    print("Rectangle area: " + str(rec_area) + ", Hull area: " + str(ch_area))
    plt.show()


if __name__ == "__main__":
    run(sys.argv[1])
