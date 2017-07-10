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
            if (j == (len(blurs) - 1)):
                continue
            tmpImage = makeDiff(blurs[j], blurs[j + 1])
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
    print('Start building octaves')
    octaves = getOctaves(img)
    print('Start diff of octaves')
    diffOctaves = getDiffOctaves(octaves)
    print('Start find key points')
    kps = findKeyPoints(diffOctaves)
    print('Remove non-pertinent key-points')
    cleanKp(kps, octaves)
    flatKps = flattenKps(kps)
    print('Build descriptor for key-points')
    descriptors = getDescriptors(octaves, flatKps)
    return descriptors, flatKps


def getDescriptors(octaves, flatKps):
    return desc.descriptor().create_descriptors(flatKps, octaves[0][0])

def getOctaves(img, nbOctaves=4, nbBlur=5, sig=1.6):
    res = [[[] for j in range(nbBlur)] for i in range(nbOctaves)]
    tmpImg = img.copy()
    for x in range(nbOctaves):
        for y in range(nbBlur):
            pass
            res[x][y] = scipy.ndimage.filters.gaussian_filter(tmpImg, sig ** (y + 1))
        tmpImg = cv2.resize(tmpImg, dsize=(0, 0), fx=1/RESIZE_COEFF, fy=1/RESIZE_COEFF)
    return res
