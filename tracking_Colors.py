# python ball_tracking.py --video ball_tracking_example.mp4  => This uses tracking on video
# python ball_tracking.py  => This uses tracking with webcam

import sys
sys.path.append('/usr/local/lib/python3.4/site-packages')

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
redLower = (128, 128, 0)
redUpper = (235, 206, 135)

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

blackLower = (102, 0, 0)
blackUpper = (255, 178, 102)

pts = deque(maxlen=args["buffer"])
pts2 = deque(maxlen=args["buffer"])
pts3 = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	(grabbed, frame2) = camera.read()
	(grabbed, frame3) = camera.read()
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	windowWidth = 500
	frame = imutils.resize(frame, width=windowWidth)
	frame2 = imutils.resize(frame2, width=windowWidth)
	frame3 = imutils.resize(frame3, width=windowWidth)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
	hsv3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, redLower, redUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	mask2 = cv2.inRange(hsv2, greenLower, greenUpper)
	mask2 = cv2.erode(mask2, None, iterations=2)
	mask2 = cv2.dilate(mask2, None, iterations=2)

	mask3 = cv2.inRange(hsv3, blackLower, blackUpper)
	mask3 = cv2.erode(mask3, None, iterations=2)
	mask3 = cv2.dilate(mask3, None, iterations=2)
	
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center2 = None

	cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center3 = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
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
			
#-
	if len(cnts2) > 0:
		c2 = max(cnts2, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c2)
		M2 = cv2.moments(c2)
		center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 0), 2)
			cv2.circle(frame, center2, 5, (255, 0, 0), -1)


	if len(cnts3) > 0:
		c3 = max(cnts3, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c3)
		M3 = cv2.moments(c3)
		center3 = (int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"]))

		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 0, 255), 2)
			cv2.circle(frame, center3, 5, (0, 255, 0), -1)	

	# update the points queue
	pts.appendleft(center)
	pts2.appendleft(center2)
	pts3.appendleft(center3)

	# loop over the set of tracked points
	#for i in xrange(1, len(pts)): => xrange has become range in python3
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
#-
	for i in range(1, len(pts2)):
		if pts2[i - 1] is None or pts2[i] is None:
			continue

		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts2[i - 1], pts2[i], (255, 0, 0), thickness)
		

	for i in range(1, len(pts3)):
		if pts3[i -1] is None or pts3[i] is None:
			continue

		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts3[i -1], pts3[i], (0, 255, 0), thickness)

	# display the (X,Y) co-ords on the frame  --Works on Pi...
	#cv2.putText(frame, "x: {}, y: {}".format(int(x), int(y)),
		#(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		#0.35, (0, 0, 255), 1)
#-	
	#cv2.putText(frame, "x: {}, y: {}".format(int(x), int(y)),
		#(300, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		#0.35, (255, 0, 0), 1)

	
	#FullScreen ?
	cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	# show the frame to our screen	
	cv2.imshow("Frame", cv2.flip(frame, 1))
	
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
