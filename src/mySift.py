import cv2
import numpy as np

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
        for j in range(1, len(img[i]) - 1):
            # res[i][j] = isMaxMin(img[i][j], [down[i - 1][j - 1], down[i][j - 1], down[i + 1][j - 1],
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
            # res[i][j - 1] = genMaxMin(diffOctaves[i][j], diffOctaves[i][j - 1], diffOctaves[i][j + 1])
            res[i][j - 1] = genMaxMin(diffOctaves[i][j], diffOctaves[i][j - 1], diffOctaves[i][j + 1])
    return res


def findCorners(img, kpList, window_size, thresh=0, k=0.05):
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]
    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = window_size/2
    #Loop through image and find our corners
    print("Finding Corners...")
    for x, y in kpList:
    # for y in range(offset, height-offset):
    #     for x in range(offset, width-offset):
            #Calculate sum of squares
        windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
        windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
        windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
        Sxx = windowIxx.sum()
        Sxy = windowIxy.sum()
        Syy = windowIyy.sum()
        #Find determinant and trace, use to get corner response
        det = (Sxx * Syy) - (Sxy**2)
        trace = Sxx + Syy
        r = det - k*(trace**2)
        #If corner response is over threshold, color the point and add to corner list
        if r > thresh:
            print(x, y, r)
            cornerList.append([x, y, r])
            color_img.itemset((y, x, 0), 0)
            color_img.itemset((y, x, 1), 0)
            color_img.itemset((y, x, 2), 255)
    pass

def cleanKp(kps, octaves):
    for i in range(len(kps)):
        for j in range(1, len(kps[i]) - 1):
            kpList = kps[i][j]
            findCorners(octaves[i][0], kpList, octaves[i][0].shape/10)


def doSift(img):
    octaves = getOctaves(img)
    diffOctaves = getDiffOctaves(octaves)
    kp = findKeyPoints(diffOctaves)
    print('Found the keypoints.')
    kp = cleanKp(kp, octaves)
    return


def print_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 800)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getOctaves(img, nbOctaves = 4, nbBlur = 5):
    res = [[[] for j in range(nbBlur)] for i in range(nbOctaves)]
    tmpImg = img.copy()
    for x in range(nbOctaves):
        tmpImg2 = tmpImg.copy()
        for y in range(nbBlur):
            pass
            tmpImg2 = cv2.blur(tmpImg2, (5, 5))
            res[x][y] = tmpImg2.copy()
            # print_image(tmpImg2)
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
