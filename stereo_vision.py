''' Proof of concept for DIY stereo vision '''

import sys
import os

import cv2
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
        elif key == ord("a"):
            print('Abort')
            sys.exit(0)

def cache_calibration_arrays(ret, mtx, dist, rvecs, tvecs, file_prefix):
    np.save(os.path.join('calibration', '{}_ret.npy'.format(file_prefix)), ret)
    np.save(os.path.join('calibration', '{}_mtx.npy'.format(file_prefix)), mtx)
    np.save(os.path.join('calibration', '{}_dist.npy'.format(file_prefix)), dist)
    np.save(os.path.join('calibration', '{}_rvecs.npy'.format(file_prefix)), rvecs)
    np.save(os.path.join('calibration', '{}_tvecs.npy'.format(file_prefix)), tvecs)

def load_calibration_arrays(file_prefix):
    try:
        ret = np.load(os.path.join('calibration', '{}_ret.npy'.format(file_prefix)))
        mtx = np.load(os.path.join('calibration', '{}_mtx.npy'.format(file_prefix)))
        dist = np.load(os.path.join('calibration', '{}_dist.npy'.format(file_prefix)))
        rvecs = np.load(os.path.join('calibration', '{}_rvecs.npy'.format(file_prefix)))
        tvecs = np.load(os.path.join('calibration', '{}_tvecs.npy'.format(file_prefix)))

        return ret, mtx, dist, rvecs, tvecs
    except IOError as e:
        print 'Could not load distortion matrices from disk...'
        return None

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

    calib_arrs = load_calibration_arrays(file_prefix)

    if calib_arrs is None:
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

        cache_calibration_arrays(ret, mtx, dist, rvecs, tvecs, file_prefix)
    else:
        print('Loaded cached arrays')
        ret, mtx, dist, rvecs, tvecs = calib_arrs

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

    return dst

def build_depth_map(left, right):

    imleft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    imright = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=11)

    disparity = stereo.compute(imleft, imright)

    plt.imshow(disparity, 'gray')
    plt.show()

    return

if __name__ == '__main__':
    left, right = capture_calibrated_shot()