import cv2
import numpy as np
import scipy.ndimage
import desc

from src.color import color
from src.harris import findCorners
from src.minMax import findKeyPoints

RESIZE_COEFF= np.sqrt(2)

def makeDiff(img1, img2):
    return cv2.absdiff(img1, img2)


def getDiffOctaves(octaves):
    res = [[[] for j in range(len(octaves[i]) - 1)] for i in range(len(octaves))]
    for i, blurs in enumerate(octaves):
        for j, _ in enumerate(blurs):
            # print("j = " + str(j) + " with maxSize: " + str(len(blurs)))
            if (j == (len(blurs) - 1)):
                continue
            tmpImage = makeDiff(blurs[j], blurs[j + 1])
            # print_image(tmpImage)
            res[i][j] = tmpImage
    return res



def cleanKp(kps, octaves):
    for i in range(len(kps)):
        for j in range(0, len(kps[i])):
            kpList = kps[i][j]
            kps[i][j] = findCorners(octaves[i][0], kpList)


def flattenKps(kps):
    res = []
    for i in range(len(kps)):
        for j in range(0, len(kps[i])):
            for kp in kps[i][j]:
                tmp = kp
                tmp[0] = int(tmp[0] * RESIZE_COEFF ** i)
                tmp[1] = int(tmp[1] * RESIZE_COEFF ** i)
                res.append(tmp)
    return res


def doSift(img):
    # source image
    octaves = getOctaves(img)
    diffOctaves = getDiffOctaves(octaves)
    kps = findKeyPoints(diffOctaves)
    print('Found the keypoints.')
    cleanKp(kps, octaves)
    flatKps = flattenKps(kps)
    descriptors = getDescriptors(octaves, flatKps)
    return descriptors, flatKps


# def doSift(img, imgRef):
#     # pattern image
#     octavesRef = getOctaves(imgRef)
#     diffOctavesRef = getDiffOctaves(octavesRef)
#     kpsRef = findKeyPoints(diffOctavesRef)
#     print('Found the keypoints.')
#     cleanKp(kpsRef, octavesRef)
#     # source image
#     octaves = getOctaves(img)
#     diffOctaves = getDiffOctaves(octaves)
#     kps = findKeyPoints(diffOctaves)
#     print('Found the keypoints.')
#     cleanKp(kps, octaves)
#     # descriptors
#     descriptorsRef = getDescriptors(octavesRef, kpsRef)
#     # descriptors
#     descriptors = getDescriptors(octaves, kps)
#     for i in range(len(descriptors)):
#         for j in range(0, len(descriptors[i])):
#             KDtree(descriptors[i][j], descriptorsRef[i][j])
#     return kps




def getDescriptors(octaves, flatKps):
    return desc.descriptor().creatDes(flatKps, octaves[0][0])




def getOctaves(img, nbOctaves=4, nbBlur=5, sig=1.6):
    res = [[[] for j in range(nbBlur)] for i in range(nbOctaves)]
    tmpImg = img.copy()
    for x in range(nbOctaves):
        for y in range(nbBlur):
            pass
            res[x][y] = scipy.ndimage.filters.gaussian_filter(tmpImg, sig ** (y + 1))
        tmpImg = cv2.resize(tmpImg, dsize=(0, 0), fx=1/RESIZE_COEFF, fy=1/RESIZE_COEFF)
    return res

# def convolution(img, mat):
#     res = img.copy()
#     matSize = len(mat)
#     matSum = numpy.sum(mat)
#
#     complicatedValue = matSize // 2
#
#     for x in range(complicatedValue, len(img) - complicatedValue):
#         for y in range(complicatedValue, len(img[x]) - complicatedValue):
#             newPixel = getNextPixel(img, x, y, mat, matSize, matSum, complicatedValue)
#             res[x][y] = newPixel
#         print("Going to " + str(x))
#     return res
#
#
# def getNextPixel(img, x, y, mat, matSize, matSum, complicatedValue):
#     r, g, b = 0, 0, 0
#     for i in range(matSize):
#         for j in range(matSize):
#             xIndex, yIndex = x + i - complicatedValue, y + j - complicatedValue
#             r = r + img[xIndex][yIndex][2] * mat[i][j]
#             g = g + img[xIndex][yIndex][1] * mat[i][j]
#             b = b + img[xIndex][yIndex][0] * mat[i][j]
#     r = r / matSum
#     g = g / matSum
#     b = b / matSum
#     return [b, g, r]
#
# def getNextPixelSafe(img, x, y, mat, matSize, matSum, complicatedValue):
#     r, g, b = 0, 0, 0
#     for i in range(matSize):
#         for j in range(matSize):
#             xIndex, yIndex = setValid(x + i - complicatedValue, y + j - complicatedValue, img)
#             r = r + img[xIndex][yIndex][2] * mat[i][j]
#             g = g + img[xIndex][yIndex][1] * mat[i][j]
#             b = b + img[xIndex][yIndex][0] * mat[i][j]
#     r = r / matSum
#     g = g / matSum
#     b = b / matSum
#     return [r, g, b]
#
#
# def setValid(x, y, img):
#     return max(min(x, len(img) - 1), 0), max(min(y, len(img[0]) - 1), 0)
