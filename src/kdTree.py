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
    print('start kd')
    for p in pDes.keys():
        x = pDes[p]
        re = tree.query(x, k=2, eps=distanceThresh, p=2, distance_upper_bound=np.inf)
        # print('similarity: ', re[0][0] / re[0][1])
        if (re[0][1] != 0 and re[0][0] / re[0][1] < similarityThresh) or re[0][1] == 0:
            pLoc = p
            sLoc = list(slocList)[re[1][0]]
            distance = re[0][0]
            if not (sLoc in sDict) or distance < result.get((sDict[sLoc], sLoc)):
                # We found a match, or a better one!
                result[(pLoc, sLoc)] = distance
                pDict[pLoc] = sLoc
                sDict[sLoc] = pLoc

    # the list of matched pixels, sorted by the distance
    finResult = sorted(result.items(), reverse=False, key=lambda d: d[1])

    # match1 = finResult[0][0]
    # match2 = finResult[1][0]
    # match3 = finResult[2][0]
    print('KD Tree comparaison is Done')
    # scalingFactor = scale.cal_factor(match1, match2, match3)
    return finResult