import cv2
import numpy as np

from src.color import color, print_image

# r_opti (11**2)/10
def findCorners(img, kpList, thresh=10000, opti_r=5, k=0.05, window_size=5, printCorner=False):
    dy, dx = np.gradient(img)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2
    height = img.shape[0]
    width = img.shape[1]
    cornerList = []
    offsetX, offsetY = window_size, window_size
    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Loop through image and find our corners
    for y, x in kpList:
        if not (x > window_size and x + 1 < width - window_size
                and y > window_size and y + 1 < height - window_size):
            continue
        windowIxx = Ixx[y - offsetY:y + offsetY + 1, x - offsetX:x + offsetX + 1]
        windowIxy = Ixy[y - offsetY:y + offsetY + 1, x - offsetX:x + offsetX + 1]
        windowIyy = Iyy[y - offsetY:y + offsetY + 1, x - offsetX:x + offsetX + 1]
        Sxx = windowIxx.sum()
        Sxy = windowIxy.sum()
        Syy = windowIyy.sum()
        # Find determinant and trace, use to get corner response
        det = (Sxx * Syy) - (Sxy ** 2)
        trace = Sxx + Syy
        r = det - k * (trace ** 2)
        # If corner response is over threshold, color the point and add to corner list

        if r > thresh:
            # print('Corner!', x, y, r, opti_r)
            cornerList.append([y, x])
            cv2.circle(img2, (x, y), 3, color('g'), thickness=-1)
        else:
            # print('Not corner :(', x, y, r)
            cv2.circle(img2, (x, y), 3, color('r'), thickness=-1)
    if printCorner:
        print_image(img2)
    return cornerList