''' Proof of concept for DIY stereo vision '''

import argparse
import sys

import cv2 as cv
import imutils

from matplotlib import pyplot as plt

def parse_args():
    ''' Parses arguments '''
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--left',
                    help='Left hand side image')
    ap.add_argument('-r', '--right',
                    help='Right hand side image')

    if len(sys.argv) <= 2:
        ap.print_help()
        sys.exit(1)

    return vars(ap.parse_args())

def get_image_paths(args):
    ''' Ensures that both a left and right image were passed '''
    l_path = args.get('left')
    r_path = args.get('right')

    if l_path is None or r_path is None:
        print('Missing argument(s)')
        sys.exit(1)

    return l_path, r_path

def run(args):
    l_path, r_path = get_image_paths(args)

    imleft = cv.imread(l_path)
    imright = cv.imread(r_path)

    imleft = imutils.resize(imleft, width=640)
    imright = imutils.resize(imright, width=640)

    imleft = cv.cvtColor(imleft, cv.COLOR_BGR2GRAY)
    imright = cv.cvtColor(imright, cv.COLOR_BGR2GRAY)

    # stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    window_size = 5
    min_disp = 32
    num_disp = 112 - min_disp
    stereo = cv.StereoSGBM(
        minDisparity=min_disp,
        numDisparities=num_disp,
        SADWindowSize=window_size,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        fullDP=False
    )
    disparity = stereo.compute(imleft, imright)

    plt.imshow(disparity, 'gray')
    plt.show()

    return

if __name__ == '__main__':
    args = parse_args()
    run(args)
