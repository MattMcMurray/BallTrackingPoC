# BallTrackingPoC
Tracking different coloured balls using computer vision

## Setting Up
1. Follow [this guide](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) to set up OpenCV on Ubuntu
2. Install the requirements in the virtual environment by running `pip install -r requirements.txt`

## Running
You will need to change:

`camera = cv2.VideoCapture(1)`

to:

`camera = cv2.VideoCapture(0)`

if you are using a built-in webcam.

### Help
`python ball_tracking.py -h` for help
