import sys
import numpy as np


def run(outfile):
    a = 2
    b = 3
    count = 0
    xs = list()
    ys = list()
    for t in np.linspace(0.01, 2*np.pi-.01, num=200):
        count += 1
        xs.append(np.cos(t-b))
        ys.append(np.sin(a*t))
    with open(outfile, 'w') as out_file:
        for row in range(len(xs)):
            out_file.write("\n%.8f\t%.8f" % (xs[row], ys[row]))


if __name__ == "__main__":
    run(sys.argv[1])
