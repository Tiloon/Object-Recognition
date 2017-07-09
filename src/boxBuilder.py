from math import sqrt, acos


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, p2):
        return sqrt((self.x + p2.x) ** 2 + (self.y + p2.y) ** 2)

    def scale(self, coeff):
        self.x *= coeff
        self.y *= coeff

    def angle(self, p2):  # en radiant
        return acos((self.x * p2.x + self.y * p2.y) / (sqrt(self.x ** 2 + self.y ** 2) * sqrt(p2.x ** 2 + p2.y ** 2)))

    def toTuple(self):
        return self.x, self.y

def findBaseBox(ps):
    x, y, x2, y2 = ps[0].x, ps[0].y, ps[0].x, ps[0].y
    for p in ps:
        x = min(p.x, x)
        x2 = max(p.x, x2)
        y = min(p.y, y)
        y2 = max(p.y, y2)
    return Point(x, y), Point(x2, y2)

