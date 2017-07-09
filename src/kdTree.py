import numpy as np
import scipy


def doKDtree(sDes, pDes, distanceThresh=0.00000000001, similarityThresh=0.90):
    tree = []
    result = {}
    # use cKD tree struture to compute the two similar pixels
    tree = scipy.spatial.cKDTree(list(sDes.values()))
    slocList = sDes.keys()
    pDict = {}
    sDict = {}
    for p in pDes.keys():
        x = pDes[p]
        re = tree.query(x, k=2, eps=distanceThresh, p=2, distance_upper_bound=np.inf)
        # print('similarity: ', re[0][0] / re[0][1])
        if re[0][1] != 0 and re[0][0] / re[0][1] < similarityThresh:
            pLoc = p
            sLoc = list(slocList)[re[1][0]]
            distance = re[0][0]
            # have not been compared before
            if not (sLoc in sDict):
                # add the result and compared pattern pixel
                # and source pixel
                result[(pLoc, sLoc)] = distance
                pDict[pLoc] = sLoc
                sDict[sLoc] = pLoc
            elif distance < result.get((sDict[sLoc], sLoc)):
                # updates the result and compared pattern pixel
                # and source pixel
                del result[(sDict[sLoc], sLoc)]
                result[(pLoc, sLoc)] = distance
                del pDict[sDict[sLoc]]
                pDict[pLoc] = sLoc
                sDict[sLoc] = pLoc
        elif re[0][1] == 0:
            pLoc = p
            sLoc = list(slocList)[re[1][0]]
            distance = re[0][0]
            # did not been compared before
            if not (sLoc in sDict):
                # add the result and compared pattern pixel
                # and source pixel
                result[(pLoc, sLoc)] = distance
                pDict[pLoc] = sLoc
                sDict[sLoc] = pLoc
            elif distance < result.get((sDict[sLoc], sLoc)):
                # updates the result and compared pattern pixel
                # and source pixel
                del result[(sDict[sLoc], sLoc)]
                result[(pLoc, sLoc)] = distance
                del pDict[sDict[sLoc]]
                pDict[pLoc] = sLoc
                sDict[sLoc] = pLoc

    # the list of matched pixels, sorted by the distance
    finResult = sorted(result.items(), reverse=False, key=lambda d: d[1])

    # match1 = finResult[0][0]
    # match2 = finResult[1][0]
    # match3 = finResult[2][0]
    print('Done')
    # scalingFactor = scale.cal_factor(match1, match2, match3)
    return finResult