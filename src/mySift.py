import cv2
import numpy as np
import scipy.ndimage
import desc

def makeDiff(img1, img2):
    return cv2.absdiff(img1, img2)


def getDiffOctaves(octaves):
    res = [[[] for j in range(len(octaves[i]) - 1)] for i in range(len(octaves))]
    for i, blurs in enumerate(octaves):
        for j, _ in enumerate(blurs):
            print("j = " + str(j) + " with maxSize: " + str(len(blurs)))
            if (j == (len(blurs) - 1)):
                continue
            tmpImage = makeDiff(blurs[j], blurs[j + 1])
            # print_image(tmpImage)
            res[i][j] = tmpImage
    return res

def isMaxMin(elt, neighbours):
    if elt > neighbours[0]:
        for x in range(1, len(neighbours)):
            if elt <= neighbours[x]:
                return 0
    elif elt < neighbours[0]:
        for x in range(1, len(neighbours)):
            if elt >= neighbours[x]:
                return 0
    else:
        return 0
    return 1

def genMaxMin(img, down, up):
    # res = np.zeros(img.shape)
    res = []
    for i in range(1, len(img) - 1):
        # for j in np.nonzero(img[i])[0]:
        for j in range(1, len(img[i]) - 1):
            tmp = isMaxMin(img[i][j], [down[i - 1][j - 1], down[i][j - 1], down[i + 1][j - 1],
                                             down[i - 1][j], down[i][j], down[i + 1][j],
                                             down[i - 1][j + 1], down[i][j + 1], down[i + 1][j + 1],
                                             img[i - 1][j - 1], img[i][j - 1], img[i + 1][j - 1],
                                             img[i - 1][j], img[i + 1][j],
                                             img[i - 1][j + 1], img[i][j + 1], img[i + 1][j + 1],
                                             up[i - 1][j - 1], up[i][j - 1], up[i + 1][j - 1],
                                             up[i - 1][j], up[i][j], up[i + 1][j],
                                             up[i - 1][j + 1], up[i][j + 1], up[i + 1][j + 1]])
            if (tmp == 1):
                res.append((i, j))
    return res

def findKeyPoints(diffOctaves):
    res = [[[] for j in range(len(diffOctaves[i]) - 2)] for i in range(len(diffOctaves))]
    for i in range(len(diffOctaves)):
        for j in range(1, len(diffOctaves[i]) - 1):
            res[i][j - 1] = genMaxMin(diffOctaves[i][j], diffOctaves[i][j - 1], diffOctaves[i][j + 1])
    return res


def findCorners(img, kpList, thresh=-1, k=0.05):
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]
    cornerList = []
    # newImg = img.copy()
    # color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offsetX, offsetY = width//20, height//20
    #Loop through image and find our corners
    print("Finding Corners...")
    for x, y in kpList:
    # for y in range(offset, height-offset):
    #     for x in range(offset, width-offset):
            #Calculate sum of squares
        if not (y-offsetY > 0 and y+offsetY+1 < height and x-offsetX > 0 and x+offsetX+1 < width):
            continue
        windowIxx = Ixx[y-offsetY:y+offsetY+1, x-offsetX:x+offsetX+1]
        windowIxy = Ixy[y-offsetY:y+offsetY+1, x-offsetX:x+offsetX+1]
        windowIyy = Iyy[y-offsetY:y+offsetY+1, x-offsetX:x+offsetX+1]
        Sxx = windowIxx.sum()
        Sxy = windowIxy.sum()
        Syy = windowIyy.sum()
        #Find determinant and trace, use to get corner response
        det = (Sxx * Syy) - (Sxy**2)
        trace = Sxx + Syy
        r = det - k*(trace**2)
        #If corner response is over threshold, color the point and add to corner list
        if r > thresh: #TODO: fix thresh value!
            print(x, y, r)
            # cornerList.append([x, y, r])
            cornerList.append([x, y])
            # color_img.itemset((y, x, 0), 0)
            # color_img.itemset((y, x, 1), 0)
            # color_img.itemset((y, x, 2), 255)
    return cornerList

def cleanKp(kps, octaves):
    for i in range(len(kps)):
        for j in range(0, len(kps[i])):
            kpList = kps[i][j]
            kps[i][j] = findCorners(octaves[i][0], kpList)


def doSift(img):
    octaves = getOctaves(img)
    diffOctaves = getDiffOctaves(octaves)
    kps = findKeyPoints(diffOctaves)
    print('Found the keypoints.')
    cleanKp(kps, octaves)
    descriptors = getDescriptors(octaves, kps, )
    return kps


def getDescriptors(octaves, kps):
    res = [[[] for j in range(len(kps[i]))] for i in range(len(kps))]
    for i in range(len(kps)):
        for j in range(0, len(kps[i])):
            res[i][j] = desc.descriptor().creatDes(kps[i][j], octaves[i][0])
    return res


def print_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 800)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getOctaves(img, nbOctaves=4, nbBlur=5, sig=1.6):
    res = [[[] for j in range(nbBlur)] for i in range(nbOctaves)]
    tmpImg = img.copy()
    for x in range(nbOctaves):
        for y in range(nbBlur):
            pass
            # tmpImg2 = cv2.blur(tmpImg2, (5, 5))
            res[x][y] = scipy.ndimage.filters.gaussian_filter(tmpImg, sig ** (y + 1))
            # print_image(res[x][y])
        tmpImg = cv2.resize(tmpImg, dsize=(0, 0), fx=0.5, fy=0.5)
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
