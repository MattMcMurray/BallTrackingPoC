''' Proof of concept for DIY stereo vision '''

import argparse
import sys

import cv2
import imutils

from matplotlib import pyplot as plt

import numpy as np

LEFT = 0
RIGHT = 1

def wait_for_q():
    while True:
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break


def calibrate_single_camera(cam, file_prefix):

    num_rows = 7
    num_cols = 5

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_cols * num_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    (grabbed, frame) = cam.read()

    if not grabbed:
        print('Could not grab')
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (num_rows, num_cols), None)

    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (num_rows, num_cols), corners, ret)
        cv2.imshow('img', frame)

        wait_for_q()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    h,  w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]

    cv2.imwrite('undistorted.png', dst)
    wait_for_q()

    cv2.destroyAllWindows()
    cam.release()


def run():
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)

    cam1.grab()
    cam2.grab()

    _, frame1 = cam1.retrieve()
    _, frame2 = cam2.retrieve()

    imleft = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    imright = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    cv2.imshow('left', imleft)
    cv2.imshow('right', imright)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cam1.release()
    cam2.release()

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=11)

    disparity = stereo.compute(imleft, imright)

    plt.imshow(disparity, 'gray')
    plt.show()

    return

if __name__ == '__main__':
    calibrate_single_camera(cv2.VideoCapture(0), 'r')
