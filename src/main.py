import sys
import cv2
import numpy as np

def main():
    print('lol')
    print(sys.argv[1])
    img = cv2.imread(sys.argv[1], 1)
    print_image(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print_image(imgray)

    im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours[4]
    #cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    print_image(img)


def applyFilter(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print_image(img)
    return img


def applyCanny(img):
    edges = cv2.Canny(img, 200, 200)
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
