import cv2
import numpy as np

from src.mySift import doSift
from src.file_picker import chooseImagePath, chooseImagePathRef, listOfPaths

from src.boxBuilder import *
from src.color import color
from src.kdTree import doKDtree
from src.payloadKps import *


def main():
    nbResize = 0
    circleSize = 5 // (nbResize + 1)
    imgPath = chooseImagePath()
    imgPathRef = chooseImagePathRef()

    # for imgPath in listOfPaths():
    #     try:
    #         print('>>>>>>>>>>>>>>>', imgPath)
    #         img, myDesc, myKps, myFinalImg = compute_or_fecth_pickle(imgPath, nbResize=nbResize, circleSize=circleSize, printImg=False)
    #     except:
    #         print('######### FAILED FOR', imgPath)
    #         continue

    print('>>>>>>>> Start Computing key-points for tested Image')
    img, myDesc, myKps, myFinalImg = compute_or_fecth_pickle(imgPath, nbResize=nbResize, circleSize=circleSize, printImg=False)
    print('>>>>>>>> Start Computing key-points for ref Image')
    imgRef, refDesc, refKps, myFinalImgRef = compute_or_fecth_pickle(imgPathRef, printImg=False)

    commonPoints = doKDtree(refDesc, myDesc)
    print('match', len(commonPoints), 'on', len(refDesc), 'proportion', len(commonPoints) / len(refDesc))

    # printing kps that matched
    refKps, imgKps = [], []
    for cp in commonPoints:
        sKp = cp[0][1]
        pKp = cp[0][0]

        y, x = sKp
        y2, x2 = pKp
        imgKps.append(Point(x2, y2))
        refKps.append(Point(x, y))

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


def myPrintKeyDiff(imgL, imgR, imgLKps, imgRKps):
    # nice print
    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]
    newimg = np.zeros((max(hL, hR), wL + wR, 3), np.uint8)
    newimg[:hL, :wL] = imgL
    newimg[:hR, wL:wR + wL] = imgR
    print('Start cleaning isolated key-point matches')
    p1, p2 = findBaseBox(imgRKps)

    imgLKps, imgRKps, center = cvCenter(imgLKps, imgRKps)
    # if len(imgLKps) < 15:
    #     print(">> No match found")

    for LKp, RKp in zip(imgLKps, imgRKps):
        x, y = RKp.toTuple()
        x2, y2 = LKp.toTuple()
        pt_imgL = (int(x2), int(y2))
        pt_imgR = (int(x + wL), int(y))
        cv2.line(newimg, pt_imgL, pt_imgR, (0, 0, 255, 100), thickness=1)

    (x, y), (x2, y2) = findBaseBox(imgRKps, tupleFormat=True)
    cv2.rectangle(newimg, (x + wL, y), (x2 + wL, y2), (0, 255, 0), thickness=15)

    sc = scale(imgLKps, imgRKps)
    # rotation(imgLKps, imgRKps)
    p1, p2 = p1.scale(sc), p2.scale(sc)
    wBox, hBox = boxSize(p1, p2)
    p1, p2 = center.add(-wBox // 2, -hBox // 2), center.add(wBox // 2, hBox // 2)
    cv2.rectangle(newimg, p1.toTuple(), p2.toTuple(), (0, 255, 0), thickness=15)

    print_image(newimg)





def compute_or_fecth_pickle(imgPath, nbResize=0, printImg=True, circleSize=5):
    img = cv2.imread(imgPath, 1)
    for i in range(nbResize):
        img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    myImg = img.copy()
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
    # pass
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 800)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
