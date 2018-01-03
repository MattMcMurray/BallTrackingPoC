from collections import deque
import argparse

import numpy as np
import imutils
import cv2

# define upper/lower bounds of colour to track
# using HSV allows better fine-tuning of colour range than RGB
HSV_LIMITS = {
    'green': {
        'lower': (71, 96, 93),
        'upper': (90, 241, 239)
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
        'lower': (106, 85, 143),
        'upper': (116, 255, 255),
    },
    'pink': {
        'lower': (159, 150, 185),
        'upper': (197, 220, 255)
    }
}

# construct argparse
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video',
                help='[Optional] path to a video file')
ap.add_argument('-b', '--buffer', type=int, default=64,
                help='max buffer size for tracked history (defaults to 64)')
args = vars(ap.parse_args())

def find_circles(frame):
    ''' 
    Uses HoughCircles to find circular objects in a video frame and returns the approximate
    centres of all found circles
    '''

    CANNY_UPPER = 120
    BLUR = 3

    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gframe = cv2.GaussianBlur(gframe, (BLUR, BLUR), 0)
    edges = cv2.Canny(gframe, CANNY_UPPER/2, CANNY_UPPER)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
                               1.1, 5, param1=CANNY_UPPER, param2=25, minRadius=5 ,maxRadius=25)


    try:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(gframe, 'Ball', (x, y), font, 4, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(gframe, (x - 1, y - 1), (x + 1, y + 1), (255, 255, 255), -1)
            cv2.circle(gframe, (x, y), r, (255, 255, 255), 1)
    except Exception as e:
        pass

    cv2.imshow('gray', gframe)
    cv2.imshow('edges', edges)

def init_masks(hsv_frame):
    '''
    Construct a mask for each colour defined in the dict
    then perform a series of dilations and erosions to remove small blobs
    left in mask.
    '''
    hsv_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)
    colour_masks = []
    colour_i = -1
    i = 0
    for colour, limits in HSV_LIMITS.iteritems():
        m = cv2.inRange(hsv_frame, limits['lower'], limits['upper'])
        m = cv2.erode(m, None, iterations=2)
        m = cv2.dilate(m, None, iterations=2)
        colour_masks.append(m)
        if colour == 'pink':
            colour_i = i
            cv2.imshow('mask', m)
        i += 1

    return colour_masks, colour_i

def run():
    ''' Main method '''
    tracker = None

    # if a video path was not supplied, grab reference to webcam
    if not args.get('video', False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args['video'])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 60, (600, 450))

    points = deque(maxlen=args.get('buffer'))
    while True:


        # grab current frame
        (grabbed, frame) = camera.read()

        if args.get('video') and not grabbed:
            break

        # resize frame, blur it, and convert it to HSV color space
        frame = imutils.resize(frame, width=600)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        colour_masks, colour_i = init_masks(hsv)

        # find contours in all masks
        contours = []
        for mask in colour_masks:
            contours.append(cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)[-2])


        if tracker is None:
            # Assume largest bounding box is ball
            padding = 5
            try:
                roi_c = max(contours[colour_i], key=cv2.contourArea)
                roi = cv2.boundingRect(roi_c)
                x, y, w, h = roi

                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x - padding, y - padding, w + padding*2, h + padding*2))
            except Exception as e:
                print(e)

        else:
            ok, bbox = tracker.update(frame)
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 255, 255), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                tracker = None


        center = None

        # only proceed if at least one contour was found
        for c in contours:
            for contour in c:
                if len(contour) > 0:
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)
                    M = cv2.moments(contour)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # only proceed if the radius meets a minimum size
                    if radius > 10:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(frame, (int(x), int(y)), int(radius),
                                (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)

                points.appendleft(center)

        # loop over the set of tracked points
        for i in xrange(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue

            # draw "history" on image
            cv2.circle(frame, points[i], 3, (0, 0, 255), -1)

        # show the frame to our screen
        cv2.imshow('Frame', frame)

        out.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    out.release()
    cv2.destroyAllWindows()
    camera.release()

if __name__ == '__main__':
    run()
