''' Proof of concept for DIY stereo vision '''

import sys
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np


TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
                        0.001)
REMAP_INTERPOLATION = cv2.INTER_LINEAR
OPTIMIZE_ALPHA = 0.25

def wait_for_q():
    while True:
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
        elif key == ord("a"):
            print 'Abort' 
            sys.exit(0)

def capture_calibrated_shot():
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    grabbed1 = cam1.grab()
    grabbed2 = cam2.grab()
    if not grabbed1 or not grabbed2:
        print('Could not grab frame for calibration')
        sys.exit(1)

    _, frame1 = cam1.retrieve()
    _, frame2 = cam2.retrieve()

    cam1.release()
    cam2.release()

    cv2.imshow('left', frame1)
    cv2.imshow('right', frame2)

    wait_for_q()

    cv2.destroyAllWindows()

    undst1 = calibrate_single_camera(frame1, 'l')
    undst2 = calibrate_single_camera(frame2, 'r')

    return undst1, undst2



def calibrate_single_camera(frame, file_prefix):

    # checkerboard inside corners
    num_rows = 7
    num_cols = 5

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_cols * num_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    print('Buidling new distortion correction matrices...')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (num_rows, num_cols), None)

    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (num_rows, num_cols), corners, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)


    h,  w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imshow('distorted_{}'.format(file_prefix), frame)
    cv2.imshow('undistorted_{}'.format(file_prefix), dst)
    wait_for_q()

    cv2.destroyAllWindows()

    return dst, objpoints, imgpoints, mtx, dist

def build_depth_map(left, right, leftMapX, leftMapY, rightMapX, rightMapY):

    stereoMatcher = cv2.StereoBM_create()

    fixedLeft = cv2.remap(left, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(right, rightMapX, rightMapY, REMAP_INTERPOLATION)

    cv2.imshow('fixedLeft', fixedLeft)
    cv2.imshow('fixedRight', fixedRight)

    wait_for_q()

    cv2.destroyAllWindows()

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    DEPTH_VISUALIZATION_SCALE = 2048
    cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)

    wait_for_q()

def stereo_calibration(left, right):
    l_im, l_objpoints, l_imgpoints, l_mtx, l_dist = left
    r_im, r_objpoints, r_imgpoints, r_mtx, r_dist = right

    # crop both images to the same dimensions
    l_h, l_w, _ = np.shape(l_im)
    r_h, r_w, _ = np.shape(r_im)
    h = min(l_h, r_h)
    w = min(l_w, r_w)
    l_im = l_im[:h, :w, :]
    r_im = r_im[:h, :w, :]

    print(np.shape(l_im))
    print(np.shape(r_im))

    cv2.imshow('croppedLeft', l_im)
    cv2.imshow('croppedRight', r_im)

    wait_for_q()

    size = (w, h)

    # TODO: snake_case

    (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        l_objpoints, l_imgpoints, r_imgpoints,
        l_mtx, l_dist,
        r_mtx, r_dist,
        size, None, None, None, None)

    (leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
        l_mtx, l_dist,
        r_mtx, r_dist,
        size, rotationMatrix, translationVector,
        None, None, None, None, None,
        cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)


    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        l_mtx, l_dist, leftRectification,
        leftProjection, size, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        r_mtx, r_dist, rightRectification,
        rightProjection, size, cv2.CV_32FC1)

    return l_im, leftMapX, leftMapY, r_im, rightMapX, rightMapY


if __name__ == '__main__':
    left, right = capture_calibrated_shot()
    l_image, leftMapX, leftMapY, r_image, rightMapX, rightMapY = stereo_calibration(left, right)

    build_depth_map(l_image, r_image, leftMapX, leftMapY, rightMapX, rightMapY)
