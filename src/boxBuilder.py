import random
from math import sqrt, acos, degrees

import itertools
from statistics import median


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, p2):
        return sqrt((self.x - p2.x) ** 2 + (self.y - p2.y) ** 2)

    def scale(self, coeff):
        return Point(int(self.x * coeff), int(self.y * coeff))

    def angle(self, p2):  # en radiant

        vectx, vecty = self.x - p2.x, self.y - p2.y
        if sqrt(vectx ** 2 + vecty ** 2) == 0:
            print('div 0')
            return 0
        print(degrees(acos(vectx / (sqrt(vectx ** 2 + vecty ** 2)))))
        return degrees(acos(vectx / (sqrt(vectx ** 2 + vecty ** 2))))
        # return degrees(acos((self.x * p2.x + self.y * p2.y) /
        #                     (sqrt(self.x ** 2 + self.y ** 2) * sqrt(p2.x ** 2 + p2.y ** 2))))

    def toTuple(self):
        return self.x, self.y

    def sub(self, p2):
        return Point(self.x - p2.x, self.y - p2.y)
    def toStr(self):
        return str(self.x) + ',' + str(self.y)

    def add(self, x, y):
        return Point(self.x + x, self.y + y)

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
        if i != j and psR[i].dist(psR[j]) != 0:
            ratios.append(psL[i].dist(psL[j]) / psR[i].dist(psR[j]))
    ratios.sort()
    if len(ratios) == 0:
        return 0
    print(median(ratios))
    # print(ratios)
    return median(ratios)

def rotation(psL, psR):
    fullComb = list(itertools.product(range(len(psL)), range(len(psL))))
    comb = [random.choice(fullComb) for _ in range(1000)]
    ratios = []
    for i, j in comb:
        if i != j:
            ratios.append((psL[i].angle(psL[j]) - psR[i].angle(psR[j])) % 180)
            print('ratio is', ratios[-1])
    ratios.sort()
    print('rotation', median(ratios))
    print('rotation', ratios)
    return median(ratios)


def findCenter(ps):
    x = 0
    y = 0
    for p in ps:
        x += p.x
        y += p.y
    return Point(x // len(ps), y // len(ps))

def cvCenter(psL, psR, delta=10):
    cPrev = Point(0, 0)
    cCurr = findCenter(psL)
    nbSuppr = len(psL) // 20
    while len(psL) > nbSuppr and cPrev.dist(cCurr) > delta:
        dists = []
        for p in psL:
            dists.append(p.dist(cCurr))
        dists2 = dists.copy()
        dists2.sort()
        maxTresh = dists2[-nbSuppr]
        psL2, psR2 = [], []
        for i, d in enumerate(dists):
            if d < maxTresh:
                psL2.append(psL[i])
                psR2.append(psR[i])
        psL, psR = psL2, psR2
        cPrev = cCurr
        cCurr = findCenter(psL)
    return psL, psR, cCurr

def boxSize(p1, p2):
    return p2.sub(p1).toTuple()
