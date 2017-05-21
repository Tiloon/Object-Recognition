import cv2


def color(code):
    if code == "r":  # or code == "red":
        return rgb(196, 47, 39)
    if code == "o":
        return rgb(232, 121, 2)
    if code == "g":
        return rgb(43, 219, 8)
    if code == "lb":
        return rgb(2, 203, 206)
    return (0, 0, 0)

def rgb(r, g, b):
    return (b, g, r)

def print_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 800)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
