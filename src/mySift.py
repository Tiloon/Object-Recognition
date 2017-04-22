import cv2


def makeDiff(img1, img2):
    return cv2.absdiff(img1, img2)


def getDiffOctaves(octaves):
    res = [[[] for j in range(len(octaves[i]))] for i in range(len(octaves))]
    for i, blurs in enumerate(octaves):
        for j, _ in enumerate(blurs):
            print("j = " + str(j) + " with maxSize: " + str(len(blurs)))
            if (j == (len(blurs) - 1)):
                continue
            tmpImage = makeDiff(blurs[j], blurs[j + 1])
            print_image(tmpImage)
            res[i][j] = tmpImage
    return res


def doSift(img):
    octaves = getOctaves(img)
    diffOctaves = getDiffOctaves(octaves)
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
