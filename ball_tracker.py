from collections import deque
import argparse

import numpy as np
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

hsv_limits = {
    'green': {
        'lower': (70, 68, 75),
        'upper': (91, 177, 255)
    },
    'yellow': {
        'lower': (33, 57, 190),
        'upper': (64, 123, 255)
    },
    'orange': {
        'lower': (0, 130, 196),
        'upper': (20, 255, 255),
    },
    'blue': {
        'lower': (101, 160, 185),
        'upper': (116, 255, 255),
    },
    'pink': {
        'lower': (159, 150, 185),
        'upper': (197, 220, 255)
    }
}

pts = deque(maxlen=args['buffer'])

# if a video path was not supplied, grab reference to webcam
if not args.get('video', False):
    camera = cv2.VideoCapture(1)
else:
    camera = cv2.VideoCapture(args['video'])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 29.4, (600, 450))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:

    # grab current frame
    (grabbed, frame) = camera.read()

    if args.get('video') and not grabbed:
        break

    # resize frame, blur it, and convert it to HSV color space
    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Construct foreground mask
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # construct a mask for each colour defined in the dict
    # then perform a series of dilations and erosions to remove small blobs
    # left in mask
    colour_masks = []
    for colour, limits in hsv_limits.iteritems():
        m = cv2.inRange(hsv, limits['lower'], limits['upper'])
        m = cv2.erode(m, None, iterations=2)
        m = cv2.dilate(m, None, iterations=2)
        colour_masks.append(m)

    # find contours in all masks
    contours = []
    for mask in colour_masks:
        contours.append(cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2])

    center = None

    # only proceed if at least one contour was found
    for cnts in contours:
        if len(cnts) > 0:
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                            (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in xrange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow('Frame', frame)
    cv2.imshow('FGMask', fgmask)

    out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

out.release()
cv2.destroyAllWindows()
camera.release()
