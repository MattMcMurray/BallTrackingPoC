from collections import deque

import numpy as np
import argparse
import imutils
import cv2

# construct argparse
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video',
                help='[Optional] path to a video file')
ap.add_argument('-b', '--buffer', type=int, default=64,
                help='max buffer size (defaults to 64)')
args = vars(ap.parse_args())

# define upper/lower bounds of colour to track
# use imutils.range-detector to determine bounds
green_lower = (18, 132, 6)
green_upper = (146, 255, 222)
pts = deque(maxlen=args['buffer'])

# if a video path was not supplied, grab reference to webcam
if not args.get('video', False):
    camera = cv2.VideoCapture(1)
else:
    camera = cv2.VideoCapture(args['video'])

while True:

    # grab current frame
    (grabbed, frame) = camera.read()

    if args.get('video') and not grabbed:
        break

    # resize frame, blur it, and convert it to HSV color space
    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color green
    # then perform a series of dilations and erosions to remove small blobs
    # left in mask
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow('original', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()
