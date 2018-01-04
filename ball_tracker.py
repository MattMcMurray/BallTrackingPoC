from collections import deque
import argparse

import imutils
import cv2

# define upper/lower bounds of colour to track
# using HSV allows better fine-tuning of colour range than RGB
HSV_LIMITS = {
    'green': {
        'lower': (70, 36, 62),
        'upper': (88, 255, 255)
    },
    'yellow': {
        'lower': (33, 65, 130),
        'upper': (50, 130, 255)
    },
    'orange': {
        'lower': (0, 145, 190),
        'upper': (15, 255, 255),
    },
    'blue': {
        'lower': (85, 105, 80),
        'upper': (165, 255, 255),
    },
    'pink': {
        'lower': (160, 105, 80),
        'upper': (180, 255, 255),
    }
}

BGR = {
    'orange': (20, 153, 255),
    'pink': (249, 54, 236),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'yellow': (48, 251, 255)
}

def _argparse():
    # construct argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video',
                    help='[Optional] path to a video file')
    ap.add_argument('-b', '--buffer', type=int, default=64,
                    help='max buffer size for tracked history (defaults to 64)')
    args = vars(ap.parse_args())

    return args

def init_mask(hsv_frame, limits):
    '''
    Construct a mask for each colour defined in the dict
    then perform a series of dilations and erosions to remove small blobs
    left in mask.
    '''
    m = cv2.inRange(hsv_frame, limits['lower'], limits['upper'])
    m = cv2.erode(m, None, iterations=2)
    m = cv2.dilate(m, None, iterations=2)

    return m

mask_hist = deque(maxlen=10)
tracker_hist = deque(maxlen=64)

def draw_history_buffer(buffer_deque, frame, color=(255, 255, 255)):
    ''' Draws a set of points detailing history of movement '''
    for i in xrange(1, len(buffer_deque)):
        if buffer_deque[i - 1] is None or buffer_deque[i] is None:
            continue

        # draw "history" on image
        cv2.circle(frame, buffer_deque[i], 3, color, -1)

def run(args):
    ''' Main method '''
    trackers = {}
    frames_since_last_seen = {}

    # if a video path was not supplied, grab reference to webcam
    if not args.get('video', False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args['video'])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 59.52, (480, 640))

    while True:
        (grabbed, frame) = camera.read()
        if args.get('video') and not grabbed:
            break

        # resize frame, blur it, and convert it to HSV color space
        frame = imutils.resize(frame, width=640)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for colour in HSV_LIMITS:

            if trackers.get(colour) is None:
                hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
                mask = init_mask(hsv, HSV_LIMITS[colour])
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[-2]

                if frames_since_last_seen.get(colour) is not None:
                    frames_since_last_seen[colour] += 1
                # Assume largest bounding box is ball
                bbox_padding = 0
                try:
                    roi_c = max(contours, key=cv2.contourArea)
                    roi = cv2.boundingRect(roi_c)
                    x, y, w, h = roi

                    mask_hist.appendleft((x + w/2, y + h/2))

                    trackers[colour] = cv2.TrackerKCF_create()
                    trackers[colour].init(frame, (x - bbox_padding, y - bbox_padding,
                                                  w + bbox_padding*2, h + bbox_padding*2))
                except Exception as e:
                    pass

            else:
                ok, bbox = trackers[colour].update(frame)
                # Draw bounding box
                if ok:
                    frames_since_last_seen[colour] = 0

                    cv2.putText(frame, colour, (int(bbox[0]), int(bbox[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    tracker_hist.appendleft((int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)))
                else:
                    frames_since_last_seen[colour] += 1
                    if frames_since_last_seen.get(colour) is not None:
                        if 5 < frames_since_last_seen[colour] <= 30:
                            trackers[colour] = None

        draw_history_buffer(mask_hist, frame, (100, 100, 255))
        draw_history_buffer(tracker_hist, frame, (0, 150, 0))

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
    args = _argparse()
    run(args)
