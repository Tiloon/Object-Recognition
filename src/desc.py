import numpy
import math


class descriptor:
    def __init__(self):
        self.size_sub_squares = 8
        self.eps = 0.00001

    def create_descriptors(self, features, imarr):
        descriptors = {}
        arr = imarr.astype(numpy.float64)
        desNum = len(features)

        for i in range(desNum):
            x, y = features[i][0], features[i][1]
            w, h = arr.shape[0], arr.shape[1]
            if x > self.size_sub_squares and x < w - 2 * self.size_sub_squares and \
                            y > self.size_sub_squares and y < h - 2 * self.size_sub_squares:
                descriptors[(x, y)] = self.create_descriptor(x, y, arr)
        return descriptors


    def gradient_properties(self, i, j, imarr):
        norm = math.sqrt((imarr[i + 1, j] - imarr[i - 1, j]) ** 2
                        + (imarr[i, j + 1] - imarr[i, j - 1]) ** 2)
        orientation = math.atan((imarr[i, j + 1] - imarr[i, j - 1])
                          / (imarr[i + 1, j] - imarr[i - 1, j] + self.eps))

        return norm, orientation


    def create_descriptor(self, i, j, imarr):
        """
        computes the 16 local area's gradient magnitude and 
        orientation around the current pixel,
        each local area contains 8 pixels
        """
        vec = [0] * 16
        vec[0] = self.localdir(i - 8, j - 8, imarr)
        vec[1] = self.localdir(i - 8, j, imarr)
        vec[2] = self.localdir(i - 8, j + 8, imarr)
        vec[3] = self.localdir(i - 8, j + 16, imarr)

        vec[4] = self.localdir(i, j - 8, imarr)
        vec[5] = self.localdir(i, j, imarr)
        vec[6] = self.localdir(i, j + 8, imarr)
        vec[7] = self.localdir(i, j + 16, imarr)

        vec[8] = self.localdir(i + 8, j - 8, imarr)
        vec[9] = self.localdir(i + 8, j, imarr)
        vec[10] = self.localdir(i + 8, j + 8, imarr)
        vec[11] = self.localdir(i + 8, j + 16, imarr)

        vec[12] = self.localdir(i + 16, j - 8, imarr)
        vec[13] = self.localdir(i + 16, j, imarr)
        vec[14] = self.localdir(i + 16, j + 8, imarr)
        vec[15] = self.localdir(i + 16, j + 16, imarr)

        return [val for subl in vec for val in subl]


    def localdir(self, i, j, imarr):
        """
        return singal pixel's direction histogram
        the histogram has 18 region
        """
        P = math.pi
        localDir = [0] * 18
        localDir2 = [0] * 18


        for b in range(i - 8, i):
            for c in range(j - 8, j):
                m, t = self.gradient_properties(b, c, imarr)


                # if t >= P * -9 / 18 and t <= P * -8 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 0)
                #     localDir[0] += m
                # if t > P * -8 / 18 and t <= P * -7 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 1)
                #     localDir[1] += m
                # if t > P * -7 / 18 and t <= P * -6 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 2)
                #     localDir[2] += m
                # if t > P * -6 / 18 and t <= P * -5 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 3)
                #     localDir[3] += m
                # if t > P * -5 / 18 and t <= P * -4 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 4)
                #     localDir[4] += m
                # if t > P * -4 / 18 and t <= P * -3 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 5)
                #     localDir[5] += m
                # if t > P * -3 / 18 and t <= P * -2 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 6)
                #     localDir[6] += m
                # if t > P * -2 / 18 and t <= P * -1 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 7)
                #     localDir[7] += m
                # if t > P * -1 / 18 and t <= 0:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 8)
                #     localDir[8] += m
                # if t > 0 and t <= P * 1 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 9)
                #     localDir[9] += m
                # if t > P * 1 / 18 and t <= P * 2 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 10)
                #     localDir[10] += m
                # if t > P * 2 / 18 and t <= P * 3 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 11)
                #     localDir[11] += m
                # if t > P * 3 / 18 and t <= P * 4 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 12)
                #     localDir[12] += m
                # if t > P * 4 / 18 and t <= P * 5 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 13)
                #     localDir[13] += m
                # if t > P * 5 / 18 and t <= P * 6 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 14)
                #     localDir[14] += m
                # if t > P * 6 / 18 and t <= P * 7 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 15)
                #     localDir[15] += m
                # if t > P * 7 / 18 and t <= P * 8 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 16)
                #     localDir[16] += m
                # if t > P * 8 / 18 and t <= P * 9 / 18:
                #     # print('real',int(round((18 * t) / P, 0)) + 8, 17)
                #     localDir[17] += m
                localDir[int(round((18 * t) / P, 0)) + 8] += m

        return localDir