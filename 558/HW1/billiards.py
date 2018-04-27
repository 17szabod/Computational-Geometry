import sys
import numpy as np
import numpy.linalg as LA
import math as mt
import matplotlib.pyplot as plt

EPS = pow(10, -8)  # constant epsilon to set


# compares x and y with safety for precision
def compare(x, y):
    if abs(x - y) < EPS:
        return 0
    elif x - y < 0:
        return 1
    else:
        return -1


# checks if a and b have the same sign
def samesign(a, b):
    if compare(a, 0) == 0 or compare(b, 0) == 0:
        return False
    return compare(a / abs(a), b / abs(b)) == 0


# returns the vector p1 - p2
def diff(p1, p2):
    ans = [0, ] * len(p1)
    for i in range(len(p1)):
        ans[i] = p1[i] - p2[i]
    return ans


# returns if p is on line
def ison(p, line):
    v1 = diff(p, line[0])
    v2 = diff(line[1], line[0])
    return compare(np.cross(v1, v2), 0) == 0


# finds where l1 intersects with the first of all possible lines, returns -1, -1 is it doesn't
def intersect(l1, *posslines):
    v1 = diff(l1[1], l1[0])
    for l2 in posslines:
        v2 = diff(l2[1], l2[0])
        if samesign(np.cross(diff(l1[0], l2[1]), v1), np.cross(diff(l1[0], l2[0]), v1)) or samesign(
                np.cross(diff(l1[1], l2[1]), v2), np.cross(diff(l1[0], l2[1]), v2)):  # segments do not intersect
            continue
        else:  # find where lines, not segments, intersect
            if ison(l1[0], l2) and not ison(l1[1], l2):  # segment started from this line (we ignore this)
                continue
            p1 = l1[0]
            p2 = l2[0]
            if v2[1] * v1[0] - v1[1] * v2[0] == 0:  # infinite solutions, ie same line (just return one of the points)
                return p2
            t = np.dot([-v1[1], v1[0]], [p1[0] - p2[0], p1[1] - p2[1]]) / (v2[1] * v1[0] - v1[1] * v2[0])
            val = p2[0] + t * v2[0], p2[1] + t * v2[1]
            return val
    return -1, -1


# reflects p1 and p2 across line and flips the segment it returns
def reflect(p1, p2, line):
    u = diff(line[1], line[0])
    v1 = diff(p1, line[1])
    v2 = diff(p2, line[1])
    w1 = np.add(line[1], np.multiply(np.dot(u, v1), np.divide(u, LA.norm(u) ** 2)))  # the foot of p1
    newp1 = np.add(p1, np.multiply(2, diff(w1, p1)))
    w2 = np.add(line[1], np.multiply(np.dot(u, v2), np.divide(u, LA.norm(u) ** 2)))  # the foot of p2
    newp2 = np.add(p2, np.multiply(2, diff(w2, p2)))
    return newp2, newp1


# draws a line with the given color (doesn't show on its own)
def drawline(p1, p2, ax, color):
    v = diff(p2, p1)
    alpha = np.linspace(0, 1)
    pointsx = np.add(p1[0], np.multiply(alpha, v[0]))
    pointsy = np.add(p1[1], np.multiply(alpha, v[1]))
    ax.plot(pointsx, pointsy, color)  # plots them in a matlab-esque way
    return


# runs through a 2d billiard instance with given parameters
def billiard(m, n, px, py, qx, qy):
    if mt.gcd(m, n) != 1:
        print("This ball will pocket early")
    fig, ax = plt.subplots()
    bot = ((m, 0), (0, 0))
    drawline(bot[0], bot[1], ax, '-b')
    top = ((0, n), (m, n))
    drawline(top[0], top[1], ax, '-b')
    left = ((0, 0), (0, n))
    drawline(left[0], left[1], ax, '-b')
    right = ((m, n), (m, 0))
    drawline(right[0], right[1], ax, '-b')
    p = (px, py)
    q = (qx, qy)
    extensionconst = ((m ** 2 + n ** 2) / 2) ** .5  # a constant to extend our lines
    count = 0
    pOld = (0, 0)
    pNew = intersect((pOld, (max(m, n),) * 2), top, right)
    if intersect((pOld, pNew), (p, q)) != (-1, -1):
        count += 1
    if ison(pNew, top):
        refover = (pNew, (pNew[0], pNew[1] - 1))
    else:
        refover = (pNew, (pNew[0] - 1, pNew[1]))
    while True:
        print(pNew)  # print the point we hit

        line = (pOld, pNew)
        if intersect(line, (p, q)) != (-1, -1):
            count += 1

        if (compare(pNew[0], 0) == 0 or compare(pNew[0], m) == 0) and (
                compare(pNew[1], 0) == 0 or compare(pNew[1], n) == 0):
            drawline(pOld, pNew, ax, '-g')  # final line will be green
            break

        drawline(pOld, pNew, ax, '-r')
        tmp = pNew

        newline = reflect(pOld, tmp, refover)

        v1 = diff(newline[1], newline[0])
        newline = newline[0], (
            np.add(newline[0], np.multiply(extensionconst, v1))[0],
            np.add(newline[0], np.multiply(extensionconst, v1))[1])
        pNew = intersect(newline, left, right, top, bot)
        if ison(pNew, top) or ison(pNew, bot):
            refover = (pNew, (pNew[0], pNew[1] - 1))
        else:
            refover = (pNew, (pNew[0] - 1, pNew[1]))
        pOld = tmp

        if pNew == (-1, -1):  # we somehow bounced out of the table
            raise Exception

    return count


# finds where l1 intersects and any of faces, or if it doesn't
def intersectpl(l1, faces):
    v1 = diff(l1[1], l1[0])
    for f in faces:
        u1 = diff(f[1], f[0])
        u2 = diff(f[2], f[0])
        if samesign(np.cross(diff(l1[0], f[0]), u1)[0], np.cross(diff(l1[1], f[0]), u1)[0]) \
                and samesign(np.cross(diff(l1[0], f[0]), u1)[1], np.cross(diff(l1[1], f[0]), u1)[1]) \
                and samesign(np.cross(diff(l1[0], f[0]), u1)[2], np.cross(diff(l1[1], f[0]), u1)[2]):
            continue  # the segment is either above or below the plane
        else:
            if isonp(l1[0], f) and not isonp(l1[1], f):  # started from this plane
                continue
            p1 = l1[0]
            a = f[0]
            M = [[u1[0], u2[0], v1[0]],
                 [u1[1], u2[1], v1[1]],
                 [u1[2], u2[2], v1[2]]]
            if LA.det(M) == 0:  # infinite solutions, inversion will fail (the line is on the plane)
                return p1
            inv = LA.inv(M)
            sol = [[p1[0] - a[0]],
                   [p1[1] - a[1]],
                   [p1[2] - a[2]]]
            t = -np.matmul(inv, sol)[2]
            val = np.add(p1, np.multiply(t, v1))
            if abs(compare(f[0][0], val[0]) - compare(val[0], f[2][0])) <= 1 \
                    and abs(compare(f[0][1], val[1]) - compare(val[1], f[2][1])) <= 1 \
                    and abs(compare(f[0][2], val[2]) - compare(val[2], f[2][2])) <= 1:  # we need f to be a rectangle
                return val
    return [-1, -1, -1]


# returns if p1 is on f
def isonp(p1, f):
    u1 = diff(f[1], f[0])
    u2 = diff(f[2], f[0])
    n = np.cross(u1, u2)
    if compare(np.dot(n, diff(f[0], p1)), 0) == 0:  # if the point is on the plane
        crosses = list()
        for i in range(1, 5):
            line = [f[i % 4], f[i - 1]]
            v = diff(line[0], line[1])
            cross = np.cross(diff(line[0], p1), v)
            crosses.append(cross)

        # returns if two arrays are equal-ish, ie the difference between any two elements is no greater than 1
        def equalarr(signs1, signs2):
            for index in range(len(signs1)):
                if abs(signs1[index] - signs2[index]) > 1:
                    return False
            return True

        # returns the signs of every element in an array with safe precisions
        def sign(arr):
            tmp = arr.copy()
            for ind in range(len(tmp)):
                if compare(tmp[ind], 0) == 0:
                    tmp[ind] = 0
                    continue
                tmp[ind] = tmp[ind] / abs(tmp[ind])
            return tmp

        return equalarr(sign(crosses[0]), sign(crosses[1])) \
               and equalarr(sign(crosses[0]), sign(crosses[2])) \
               and equalarr(sign(crosses[0]),
                            sign(crosses[3]))  # uncomfortably makes sure the crossproducts all have the same sign
    else:
        return False


# plays a game of 3d billiards with the given problem instance, counting the number of times it crosses the face pqrs
def threedbilliard(m, n, o, p, q, r, s):
    bounceCount = 0
    st = [0, 0, 0]
    if mt.gcd(m, n) != 1 or mt.gcd(m, o) != 1 or mt.gcd(o, n) != 1:
        print("This ball will pocket early")
    faces = [[[0, 0, 0], [m, 0, 0], [m, n, 0], [0, n, 0]],
             [[0, 0, 0], [0, 0, o], [0, n, o], [0, n, 0]],
             [[0, 0, 0], [m, 0, 0], [m, 0, o], [0, 0, o]],
             [[m, n, o], [m, 0, o], [0, 0, o], [0, n, o]],
             [[m, n, o], [m, n, 0], [0, n, 0], [0, n, o]],
             [[m, n, o], [m, n, 0], [m, 0, 0], [m, 0, o]]]

    extensionconst = ((m ** 2 + n ** 2 + o ** 2) / 3) ** .5  # a constant to extend every line with
    count = 0
    pOld = st
    pNew = intersectpl((pOld, [max(m, n, o), ] * 3), faces)
    if intersectpl((pOld, pNew), [[p, q, r, s]]) != [-1, -1, -1]:
        count += 1
    foundfaces = list()
    for f in faces:
        if isonp(pNew, f):
            if len(foundfaces) > 0:  # we hit an edge
                f1 = foundfaces[0]
                f2 = f.copy()
                f = list()  # the face we would put to bounce over
                for point in f1:
                    if not f2.__contains__(point):
                        f.append(point)
                for point in f2:
                    if not f1.__contains__(point):
                        f.append(point)
                normal = np.cross(diff(f[1], f[0]), diff(f[2], f[0]))
                normalpoint = np.add(pNew, np.sign(normal))  # normalize this normal curve, ie make it 1, -1, and 0s
                refover = (pNew, normalpoint)
                break
            normal = np.cross(diff(f[1], f[0]), diff(f[2], f[0]))
            normalpoint = np.add(pNew, normal)
            refover = (pNew, normalpoint)
            foundfaces.append(f)
    while True:
        bounceCount = bounceCount + 1;
        print(pNew)

        line = (pOld, pNew)
        if np.array_equal(intersectpl(line, [(p, q, r, s)]), [-1, -1, -1]):
            count += 1
        xgood = compare(pNew[0], 0) == 0 or compare(pNew[0], m) == 0
        ygood = compare(pNew[1], 0) == 0 or compare(pNew[1], n) == 0
        zgood = compare(pNew[2], 0) == 0 or compare(pNew[2], o) == 0
        if xgood and ygood and zgood:  # hit a pocket
            break
        tmp = pNew

        newline = reflect(pOld, tmp, refover)

        v1 = diff(newline[1], newline[0])
        if v1[0] == 0 and v1[1] == 0:
            raise Exception("Stopped moving")
        newline = newline[0], (np.add(newline[0], np.multiply(extensionconst, v1)))  # extend the line
        pNew = intersectpl(newline, faces)  # find the new point
        foundfaces = list()
        for f in faces:  # same as above
            if isonp(pNew, f):
                if len(foundfaces) > 0:
                    f1 = foundfaces[0]
                    f2 = f.copy()
                    f = list()
                    for point in f1:
                        if not f2.__contains__(point):
                            f.append(point)
                    for point in f2:
                        if not f1.__contains__(point):
                            f.append(point)
                    normal = np.cross(diff(f[1], f[0]), diff(f[2], f[0]))
                    normalpoint = np.add(pNew, np.sign(normal))
                    refover = (pNew, normalpoint)
                    break
                normal = np.cross(diff(f[1], f[0]), diff(f[2], f[0]))
                normalpoint = np.add(pNew, normal)
                refover = (pNew, normalpoint)
                foundfaces.append(f)

        pOld = tmp

        if np.array_equal(pNew, [-1, -1, -1]):
            raise Exception("Out of bounds")
    print(str(bounceCount) + " should equal " + str(m*n+m*o+n*o-mt.gcd(m,n)*o-mt.gcd(o,n)*m-mt.gcd(m,o)*n))
    return count


if __name__ == "__main__":
    args = sys.argv
    # example args for 3d: "3d m n o"
    if str(args[1]) == "3d":
        print(threedbilliard(int(str(args[2])), int(str(args[3])), int(str(args[4])),
                             [0, 4, 0], [0, 4, 3], [5, 0, 0], [5, 0, 3]))
        # the points p q r s need to be edited manually if you want them changed
    # example args for 2d: "m n px py qx qy"
    else:
        print(billiard(int(str(args[1])), int(str(args[2])), float(str(args[3])), float(str(args[4])),
                       float(str(args[5])),
                       float(str(args[6]))))
        plt.show()
