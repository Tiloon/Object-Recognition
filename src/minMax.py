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
    res = []
    for i in range(1, len(img) - 1):
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