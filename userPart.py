#for neural net part
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from realModelTrain2 import *

#for openCV part
import cv2
import imutils
import time
import random

def getImageModelPred():
	#load model
	model = load_model("model.h5")
	img = ["userPicture.jpg"]
	X, y = read_and_process_images("", img)
	x = np.array(X)
	test_datagen = ImageDataGenerator(rescale=1./255)
	for batch in test_datagen.flow(x, batch_size=1):
		pred = model.predict(batch)
		print(pred)
	return pred

def getSoundModelPred():
	return 42

#
def avgPreds():
	predImg = getImageModelPred()
	predSound = getSoundModelPred()
	#avg them somehow, get chord prediction??
	return 42

chords = ['aMajor','aMinor','bMinor','cMajor','dMajor',
		  'dMinor','eMajor','eMinor','fMajor','eMinor',
		  'gMajor']
random.shuffle(chords)

def run():
	camera = cv2.VideoCapture(0)
	oldtime = time.time()
	secondsDisplay = time.time()
	seconds = 5
	i=0
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

		#5 seconds have passed, display next chord
		if (time.time() - oldtime > 5):
			i+=1
			i = i % len(chords)
			oldtime = time.time() - 3
			secondsDisplay = time.time()
			seconds = 5
			#take picture of roi box after 5 seconds
			cv2.imwrite('userPicture' + '.jpg', roi)
			######## NEED TO ADD THIS CODE ##############
			#run neural networks to see if it matches the correct one
			answer = 'gMajor'
			if (answer == chords[i-1]):
				cv2.putText(clone, "SUCCESS", (200,200),
							cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
			else:
				cv2.putText(clone, "FAIL", (200,200),
						cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))

		#draw chord that user needs to play
		cv2.putText(clone, chords[i], (450, 150), 
			cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))

		#have seconds go down
		if (time.time() - secondsDisplay >= 1):
			secondsDisplay = time.time()
			seconds -= 1

		#draw number of seconds
		cv2.putText(clone, str(seconds), (350, 300), 
			cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))

		# display the frame with roi box
		cv2.imshow("Video Feed", clone)

		#after 20 chords, display your score and the chords you got correct
		if i > 20:
			break

		#if you pressed 'q' then you quit
		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord('q'):
			break
run()

