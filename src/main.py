import cv2
import numpy as np

from mySift import doSift
from file_picker import chooseImagePath, chooseImagePathRef

from src.boxBuilder import Point, findBaseBox
from src.color import color
from src.kdTree import doKDtree
from src.payloadKps import *


def main():
    nbResize = 0
    circleSize = 5 // (nbResize + 1)
    imgPath = chooseImagePath()
    imgPathRef = chooseImagePathRef()

    img, myDesc, myKps, myFinalImg = compute_or_fecth_pickle(imgPath, nbResize=nbResize, circleSize=circleSize, printImg=False)
    imgRef, refDesc, refKps, myFinalImgRef = compute_or_fecth_pickle(imgPathRef, printImg=False)


    # matching kps
    # commonPoints = doKDtree(myDesc, refDesc)
    commonPoints = doKDtree(refDesc, myDesc)
    print('match', len(commonPoints), 'on', len(refDesc), 'proportion', len(commonPoints) / len(refDesc))

    # printing kps that matched
    refKps, imgKps = [], []
    for cp in commonPoints:
        sKp = cp[0][1]
        pKp = cp[0][0]
        precision = cp[1]  # TODO: can this precision be useful?
        # print(precision)
        y, x = sKp
        imgKps.append(Point(x, y))
        y2, x2 = pKp
        refKps.append(Point(x2, y2))

        cv2.circle(myFinalImg, (x, y), 5, color('g'), thickness=-1)
        cv2.circle(myFinalImgRef, (x2, y2), 5, color('g'), thickness=-1)



    myPrintKeyDiff(img, imgRef, imgKps, refKps)

    # tryOCVSift(img, imgRef, myFinalImg)
    return


def tryOCVSift(img, imgRef, myFinalImg):
    sift = cv2.xfeatures2d.SIFT_create()
    (skp, sd) = sift.detectAndCompute(img, None)
    print("#IMG kps: {}, descriptors: {}".format(len(skp), sd.shape))
    (tkp, td) = sift.detectAndCompute(imgRef, None)
    print("#REF  kps: {}, descriptors: {}".format(len(tkp), td.shape))
    # cv2.drawKeypoints(imgRef, tkp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(myFinalImg, skp, myFinalImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print_image(myFinalImg)


def myPrintKeyDiff(imgL, imgR, imgLKps,imgRKps):
    # nice print
    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]
    newimg = np.zeros((max(hL, hR), wL + wR, 3), np.uint8)
    newimg[:hL, :wL] = imgL
    newimg[:hR, wL:wR + wL] = imgR
    for LKp, RKp in zip(imgLKps, imgRKps):
        x, y = LKp.toTuple()
        x2, y2 = RKp.toTuple()
        pt_imgL = (int(x2), int(y2))
        pt_imgR = (int(x + wL), int(y))
        cv2.line(newimg, pt_imgL, pt_imgR, (0, 0, 255))
    # b1, b2 = findBaseBox()
    print_image(newimg)


def compute_or_fecth_pickle(imgPath, nbResize=0, printImg=True, circleSize=5):
    img = cv2.imread(imgPath, 1)
    for i in range(nbResize):
        img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    myImg = img.copy()
    # imgGray = cv2.cvtColor(myImg, cv2.COLOR_BGR2GRAY)  # PASSER EN NIVEAU DE VERT ?
    imgGray = myImg[:, :, 1]

    myDesc, myKps = None, None
    if pickleExist(imgPath):
        print("Using pre-computed key-points")
        myDesc, myKps = loadPickle(imgPath)
    else:
        print("No precomputed key-points. Start computing key-points")
        myDesc, myKps = doSift(imgGray)
        savePickle(payloadKps(myDesc, myKps), imgPath)

    myFinalImg = img.copy()

    if printImg:
        for kp in myKps:
            y, x = kp[0], kp[1]
            cv2.circle(myFinalImg, (x, y), circleSize, color('lb'), thickness=-1)
        print_image(myFinalImg)

    return img, myDesc, myKps, myFinalImg


def print_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 800)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
