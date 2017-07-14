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
    return 0, 0, 0


def rgb(r, g, b):
    return b, g, r

def print_image(img, name='image'):
    # pass
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 800)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_images(imgs, resize=True, name='image'):
    # pass
    for i, img in enumerate(imgs):
        cv2.namedWindow(name + str(i), cv2.WINDOW_NORMAL)
        if resize:
            cv2.resizeWindow(name + str(i), 600, 800)
        cv2.imshow(name + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()