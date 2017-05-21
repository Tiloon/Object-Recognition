import cv2
import numpy as np

from mySift import doSift, doKDtree
from file_picker import *

from src.color import color


def main():
    nbResize = 3
    circleSize = 5 // (nbResize + 1)
    imgPath = chooseImagePath()
    imgPathRef = chooseImagePathRef()

    img = cv2.imread(imgPath, 1)
    for i in range(nbResize):
        img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    myImg = img.copy()
    # imgGray = cv2.cvtColor(myImg, cv2.COLOR_BGR2GRAY)  # PASSER EN NIVEAU DE VERT ?
    imgGray = myImg[:, :, 1]
    myDesc, myKps = doSift(imgGray)

    imgRef = cv2.imread(imgPathRef, 1)
    for i in range(nbResize):
        imgRef = cv2.resize(imgRef, dsize=(0, 0), fx=0.5, fy=0.5)
    myImgRef = imgRef.copy()
    nbResize = 0
    # imgGrayRef = cv2.cvtColor(myImgRef, cv2.COLOR_BGR2GRAY)
    imgGrayRef = myImgRef[:, :, 1]
    refDesc, refKps = doSift(imgGrayRef)

    # Adding kps to images in color light blue
    myFinalImgRef = imgRef.copy()
    for kp in refKps:
        y, x = kp[0], kp[1]
        cv2.circle(myFinalImgRef, (x, y), circleSize, color('lb'), thickness=-1)
    print_image(myFinalImgRef)
    myFinalImg = img.copy()
    for kp in myKps:
        y, x = kp[0], kp[1]
        cv2.circle(myFinalImg, (x, y), circleSize, color('lb'), thickness=-1)
    print_image(myFinalImg)

    # matching kps
    commonPoints = doKDtree(myDesc, refDesc)
    print('proportion', len(commonPoints) / len(refDesc))

    # printing kps that matched
    for cp in commonPoints:
        sKp = cp[0][1]
        pKp = cp[0][0]
        precision = cp[1]  # TODO: can this precision be useful?
        print(precision)
        y, x = sKp
        y2, x2 = pKp
        cv2.circle(myFinalImg, (x, y), 5, color('r'), thickness=-1)
        cv2.circle(myFinalImgRef, (x2, y2), 5, color('r'), thickness=-1)

    myPrintKeyDiff(img, imgRef, commonPoints)
    # myFindBox(myFinalImg, skp_common)

    # print_image(imgRef)
    # return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print_image(gray)
    sift = cv2.xfeatures2d.SIFT_create()
    (skp, sd) = sift.detectAndCompute(img, None)
    print("#IMG kps: {}, descriptors: {}".format(len(skp), sd.shape))
    (tkp, td) = sift.detectAndCompute(imgRef, None)
    print("#REF  kps: {}, descriptors: {}".format(len(tkp), td.shape))
    # cv2.drawKeypoints(imgRef, tkp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(myFinalImg, skp, myFinalImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print_image(myFinalImg)
    return

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann.Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = list(range(len(dist)))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]

    skp_final = []
    for i, dis in zip(idx, dist):
        if dis < 10:
            skp_final.append(skp[i])
        else:
            break

    print("#FINAL kps: {}".format(len(skp_final)))
    cv2.drawKeypoints(imgRef, tkp, imgRef, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print_image(imgRef)
    cv2.drawKeypoints(gray, skp_final, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print_image(img)

    printKeyDiff(img, imgRef, skp_final, tkp)

    findBox(img, skp_final)


    # lower_white = np.array([0, 0, 0])
    # upper_white = np.array([0, 0, 255])
    # myMask2 = cv2.inRange(imgHSV, lower_white, upper_white)
    # print_image(myMask2)
    # TODO: dans le mask, dilater + erosion pour trouver les carres noir
    # foutre des marqueurs dans les carres trouves
    # avec signature chercher les ecriture Minute maid? (chaud car arrondis)


    # kernel = np.ones((3,3), np.uint8)
    # Faire du close est une mauvaise idee, on va juste close entre le POMME en dessous et le rectangle
    # n = 6
    # myMaskEr = cv2.erode(myMask, kernel, iterations=n)
    # print_image(myMaskEr)
    # myMaskEr = cv2.dilate(myMaskEr, kernel, iterations=n)
    # print_image(myMaskEr)
    # res = cv2.bitwise_and(myMask, myMask, mask=myMaskEr)
    # print_image(res)


# def saveRef():
#     imgRefPath = chooseImagePathRef()
#     imgRef = cv2.imread(imgRefPath, 1)
#     gray = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
#     print_image(gray)
#     sift = cv2.xfeatures2d.SIFT_create()
#     (skp, sd) = sift.detectAndCompute(imgRef, None)
#
#     index = []
#     for point in skp:
#         temp = (point.pt, point.size, point.angle, point.response, point.octave,
#                 point.class_id)
#         index.append(temp)
#
#     # Dump the keypoints
#     f = open("keypoints.txt", "w")
#     f.write(pickle.dumps(index))
#     f.close()
#
# def loadRef():
#     # TODO: fix this
#     index = pickle.loads(open("keypoints.txt").read())
#
#     kp = []
#
#     for point in index:
#         temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
#                             _response=point[3], _octave=point[4], _class_id=point[5])
#         kp.append(temp)
#
#     return kp

def myFindBox(img, skp_final):
    xCenter, yCenter = myFindCenter(skp_final)
    img2 = img.copy()
    cv2.circle(img2, (xCenter, yCenter), 10, color('lb'), thickness=-1)
    print_image(img2)
    moyDist = 0
    for kp in skp_final:
        moyDist += np.math.sqrt((kp[0] - xCenter) ** 2 + (kp[1] - yCenter) ** 2)
    moyDist = int(moyDist // len(skp_final))
    skp_final_proper = skp_final.copy()
    for kp in skp_final:
        dist = np.math.sqrt((kp[0] - xCenter) ** 2 + (kp[1] - yCenter) ** 2)
        if (dist < moyDist):
            cv2.circle(img2, (int(kp[0]), int(kp[1])), 10, color('g'), thickness=2)
        if (dist < moyDist * 2):
            cv2.circle(img2, (int(kp[0]), int(kp[1])), 10, color('o'), thickness=2)
        else:
            skp_final_proper.remove(kp)
            cv2.circle(img2, (int(kp[0]), int(kp[1])), 10, color('r'), thickness=-1)

    print_image(img2)

    xCenter, yCenter = myFindCenter(skp_final_proper)

    img3 = img.copy()
    cv2.circle(img3, (xCenter, yCenter), 10, color('lb'), thickness=-1)
    print_image(img3)

    if len(skp_final_proper) > 0:
        maxX, maxY, minX, minY = skp_final_proper[0][0], skp_final_proper[0][1], skp_final_proper[0][0], \
                                 skp_final_proper[0][1]
        for kp in skp_final_proper:
            maxX = max(maxX, kp[0])
            maxY = max(maxY, kp[1])
            minX = min(minX, kp[0])
            minY = min(minY, kp[1])

        cv2.rectangle(img3, (int(maxX), int(maxY)), (int(minX), int(minY)), color('o'), thickness=5)

        w, h = maxX - minX, maxY - minY

        if h > w:
            print("swaping")
            maxX, minX, maxY, minY = maxY, minY, maxX, minX
            w, h = maxX - minX, maxY - minY

        maxX, minX = maxX + w, minX - w
        maxY, minY = maxY + 5 * h, minY - 2 * h

        cv2.rectangle(img3, (int(maxX), int(maxY)), (int(minX), int(minY)), color('r'), thickness=5)
        print_image(img3)


def findBox(img, skp_final):
    xCenter, yCenter = findCenter(skp_final)
    img2 = img.copy()
    cv2.circle(img2, (xCenter, yCenter), 10, color('lb'), thickness=-1)
    print_image(img2)
    moyDist = 0
    for kp in skp_final:
        moyDist += np.math.sqrt((kp.pt[0] - xCenter) ** 2 + (kp.pt[1] - yCenter) ** 2)
    moyDist = int(moyDist // len(skp_final))
    skp_final_proper = skp_final.copy()
    for kp in skp_final:
        dist = np.math.sqrt((kp.pt[0] - xCenter) ** 2 + (kp.pt[1] - yCenter) ** 2)
        if (dist < moyDist):
            cv2.circle(img2, (int(kp.pt[0]), int(kp.pt[1])), 10, color('g'), thickness=2)
        if (dist < moyDist * 2):
            cv2.circle(img2, (int(kp.pt[0]), int(kp.pt[1])), 10, color('o'), thickness=2)
        else:
            skp_final_proper.remove(kp)
            cv2.circle(img2, (int(kp.pt[0]), int(kp.pt[1])), 10, color('r'), thickness=-1)

    print_image(img2)

    xCenter, yCenter = findCenter(skp_final_proper)

    img3 = img.copy()
    cv2.circle(img3, (xCenter, yCenter), 10, color('lb'), thickness=-1)
    print_image(img3)

    if len(skp_final_proper) > 0:
        maxX, maxY, minX, minY = skp_final_proper[0].pt[0], skp_final_proper[0].pt[1], skp_final_proper[0].pt[0], \
                                 skp_final_proper[0].pt[1]
        for kp in skp_final_proper:
            maxX = max(maxX, kp.pt[0])
            maxY = max(maxY, kp.pt[1])
            minX = min(minX, kp.pt[0])
            minY = min(minY, kp.pt[1])

        cv2.rectangle(img3, (int(maxX), int(maxY)), (int(minX), int(minY)), color('o'), thickness=5)

        w, h = maxX - minX, maxY - minY

        if h > w:
            print("swaping")
            maxX, minX, maxY, minY = maxY, minY, maxX, minX
            w, h = maxX - minX, maxY - minY

        maxX, minX = maxX + w, minX - w
        maxY, minY = maxY + 5 * h, minY - 2 * h

        cv2.rectangle(img3, (int(maxX), int(maxY)), (int(minX), int(minY)), color('r'), thickness=5)
        print_image(img3)


def myFindCenter(skp_final):
    (xCenter, yCenter) = (0, 0)
    for kp in skp_final:
        (xCenter, yCenter) = (xCenter + kp[0], yCenter + kp[1])
    (xCenter, yCenter) = (int(xCenter // len(skp_final)), int(yCenter // len(skp_final)))
    print((xCenter, yCenter))
    return xCenter, yCenter


def findCenter(skp_final):
    (xCenter, yCenter) = (0, 0)
    for kp in skp_final:
        (xCenter, yCenter) = (xCenter + kp.pt[0], yCenter + kp.pt[1])
    (xCenter, yCenter) = (int(xCenter // len(skp_final)), int(yCenter // len(skp_final)))
    print((xCenter, yCenter))
    return xCenter, yCenter







def myPrintKeyDiff(img, imgRef, cps):
    # nice print
    h1, w1 = img.shape[:2]
    h2, w2 = imgRef.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[:h2, :w2] = imgRef
    newimg[:h1, w2:w1 + w2] = img
    for cp in cps:
        pKp = cp[0][0]
        sKp = cp[0][1]
        y, x = sKp
        y2, x2 = pKp
        pt_a = (int(x2), int(y2))
        pt_b = (int(x + w2), int(y))
        cv2.line(newimg, pt_a, pt_b, (0, 0, 255))
    print_image(newimg)


def printKeyDiff(img, imgRef, skp_final, tkp):
    # nice print
    h1, w1 = img.shape[:2]
    h2, w2 = imgRef.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[:h2, :w2] = imgRef
    newimg[:h1, w2:w1 + w2] = img
    tkp = tkp
    skp = skp_final
    for i in range(min(len(tkp), len(skp))):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]))
        pt_b = (int(skp[i].pt[0] + w2), int(skp[i].pt[1]))
        cv2.line(newimg, pt_a, pt_b, (0, 0, 255))
    print_image(newimg)


def randomTry(img):
    # imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print_image(imgGray)
    # imgGray = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2RGB)
    # print_image(imgGray)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print_image(imgHSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    myMask = cv2.inRange(imgHSV, lower_black, upper_black)
    print_image(myMask)




def applyFilter(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print_image(img)
    return img


def applyCanny(img):
    edges = cv2.Canny(img, 200, 100)
    print_image(edges)
    return edges


def applyHough(edges, img):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print_image(img)


def print_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 800)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
