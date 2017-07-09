import random
from math import sqrt, acos

import itertools
from statistics import median


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, p2):
        return sqrt((self.x + p2.x) ** 2 + (self.y + p2.y) ** 2)

    def scale(self, coeff):
        return Point(int(self.x * coeff), int(self.y * coeff))

    def angle(self, p2):  # en radiant
        return acos((self.x * p2.x + self.y * p2.y) / (sqrt(self.x ** 2 + self.y ** 2) * sqrt(p2.x ** 2 + p2.y ** 2)))

    def toTuple(self):
        return self.x, self.y

def findBaseBox(ps, tuple=False):
    x, y, x2, y2 = ps[0].x, ps[0].y, ps[0].x, ps[0].y
    for p in ps:
        x = min(p.x, x)
        x2 = max(p.x, x2)
        y = min(p.y, y)
        y2 = max(p.y, y2)
    if tuple:
        return Point(x, y).toTuple(), Point(x2, y2).toTuple()
    else:
        return Point(x, y), Point(x2, y2)

def scale(psL, psR):
    fullComb = list(itertools.product(range(len(psL)), range(len(psL))))
    comb = [random.choice(fullComb) for _ in range(1000)]
    ratios = []
    for i, j in comb:
        # print(psR[i].toTuple(), psR[j].toTuple(), psR[i].dist(psR[j]), psL[i].dist(psL[j]))
        # ratios.append(psR[i].dist(psR[j]) / psL[i].dist(psL[j]))
        ratios.append(psL[i].dist(psL[j]) / psR[i].dist(psR[j]) )
    print(median(ratios))
    # print(ratios)
    return median(ratios)
