import cv2
import time   
import imutils
import numpy as np


#change name of this fxn at some point
def findHand():
	camera = cv2.VideoCapture(0)
	counter=51
	while (True):
		# region of interest (ROI) coordinates
		top, right, bottom, left = 200, 0, 450, 250
		#would usually have a while true here, for actual user will have
		#for your purposes it is not there
		# get the current frame
		(success, frame) = camera.read()
		# resize the frame
		frame = imutils.resize(frame, width=700)
		# flip the frame so that it is not mirror view
		frame = cv2.flip(frame, 1)
		# copy the frame
		clone = frame.copy()
		# get the height and width of the frame
		(height, width) = frame.shape[:2]
		# get the ROI
		roi = frame[top:bottom, right:left]
		# draw roi box
		cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
		# display the frame with roi box
		cv2.imshow("Video Feed", clone)
		keypress = cv2.waitKey(1) & 0xFF
	    #if you pressed 'y' then the thing will take a picture and show you
		if keypress == ord('y'):
			cv2.imshow("Image", roi)
			keypress = cv2.waitKey(300) & 0xFF
			#if you press y again it will ask you for the name of the image and then save it
			if keypress == ord('y'):
				cv2.imwrite('gMajor' + str(counter) + '.jpg', roi)
				counter=counter+1
		if keypress == ord('q'):
			break
findHand()

'''
#allows user to take pictures (ROI), for training purposes only
def getPictureData():
	camera = cv2.VideoCapture(0)
	#coordinates for ROI 
	top, right, bottom, left = 200, 0, 450, 250
	while(True):
		# if you press "q", program will quit
		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord("q"):
			break
		(success, frame) = camera.read()
		# resize the frame
        frame = imutils.resize(frame, width=700)
        # flip the frame so that it is not mirror view
        frame = cv2.flip(frame, 1)
        # get the ROI
        roi = frame[top:bottom, right:left]
        time.sleep(3)
        cv2.imshow("Image", roi)
        #if the image is good, then the computer will save it
        val = input("Good image? Enter y or n")
        if (val == 'y'):
        	#you can specify the name of the image
        	val = input("Enter name of image")
        	cv2.imwrite(val+'.jpg', roi)
	camera.release()   

getPictureData()
'''
'''
if for some reason we want to figure out the exact position of the hand, here is the code to do that
'''
# global variables
bg = None

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return
    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def findHand():
    #weight for running average
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    # region of interest (ROI) coordinates
    top, right, bottom, left = 200, 0, 450, 250
    num_frames = 0

    while(True):
        # get the current frame
        (success, frame) = camera.read()
        # resize the frame
        frame = imutils.resize(frame, width=700)
        # flip the frame so that it is not mirror view
        frame = cv2.flip(frame, 1)
        # copy the frame
        clone = frame.copy()
        # get the height and width of the frame
        (height, width) = frame.shape[:2]
        # get the ROI
        roi = frame[top:bottom, right:left]

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # if the user pressed "q", then stop looping
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
'''
#clean up
camera.release()
cv2.destroyAllWindows()
'''

