import numpy
import math


class descriptor:
    def __init__(self):
        self.size_sub_squares = 8
        self.eps = 0.00001

    def create_descriptors(self, features, img):
        descriptors = {}
        floatImg = img.astype(numpy.float64)
        desNum = len(features)

        for i in range(desNum):
            x, y = features[i][0], features[i][1]
            w, h = img.shape[0], img.shape[1]
            if self.size_sub_squares < x < w - 2 * self.size_sub_squares \
                    and self.size_sub_squares < y < h - 2 * self.size_sub_squares:
                descriptors[(x, y)] = self.create_descriptor(x, y, floatImg)
        return descriptors

    def create_descriptor(self, x, y, img):
        hists = [self.gradHist(x - 8, y - 8, img),
                 self.gradHist(x - 8, y, img),
                 self.gradHist(x - 8, y + 8, img),
                 self.gradHist(x - 8, y + 16, img),
                 self.gradHist(x, y - 8, img),
                 self.gradHist(x, y, img),
                 self.gradHist(x, y + 8, img),
                 self.gradHist(x, y + 16, img),
                 self.gradHist(x + 8, y - 8, img),
                 self.gradHist(x + 8, y, img),
                 self.gradHist(x + 8, y + 8, img),
                 self.gradHist(x + 8, y + 16, img),
                 self.gradHist(x + 16, y - 8, img),
                 self.gradHist(x + 16, y, img),
                 self.gradHist(x + 16, y + 8, img),
                 self.gradHist(x + 16, y + 16, img)]
        return [col for hist in hists for col in hist]  # group hists by values

    def gradHist(self, x, y, img):
        P = math.pi
        localDir = [0] * 18

        for b in range(x - 8, x):
            for c in range(y - 8, y):
                m, t = self.gradient_properties(b, c, img)
                localDir[int(round((18 * t) / P, 0)) + 8] += m
        return localDir


    def gradient_properties(self, x, y, img):
        norm = math.sqrt((img[x + 1, y] - img[x - 1, y]) ** 2 + (img[x, y + 1] - img[x, y - 1]) ** 2)
        orientation = math.atan((img[x, y + 1] - img[x, y - 1]) / (img[x + 1, y] - img[x - 1, y] + self.eps))
        return norm, orientation