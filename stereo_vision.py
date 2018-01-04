''' Proof of concept for DIY stereo vision '''

import argparse
import sys

import cv2 as cv
import imutils

from matplotlib import pyplot as plt

LEFT = 0
RIGHT = 1

def calibrate_single_camera(cam):
    (grabbed, frame) = cam.read()

    cv.imshow('frame', frame)

    while True:
        key = cv.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    cv.destroyAllWindows()
    cam.release()


def run():
    cam1 = cv.VideoCapture(0)
    cam2 = cv.VideoCapture(1)

    cam1.grab()
    cam2.grab()

    _, frame1 = cam1.retrieve()
    _, frame2 = cam2.retrieve()

    imleft = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    imright = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    cv.imshow('left', imleft)
    cv.imshow('right', imright)

    while True:
        key = cv.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    cv.destroyAllWindows()
    cam1.release()
    cam2.release()

    stereo = cv.StereoBM_create(numDisparities=32, blockSize=11)

    disparity = stereo.compute(imleft, imright)

    plt.imshow(disparity, 'gray')
    plt.show()

    return

if __name__ == '__main__':
    calibrate_single_camera(cv.VideoCapture(0))
