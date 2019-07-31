#for neural net part
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import operator

#for openCV part
import cv2
import imutils
import time
import random

#for audio recording
import pyaudio
import wave

#for audio to frequency conversion
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from os import walk
from scipy.fftpack import fft, fftfreq
from PIL import Image

#variables for audio recording
CHUNK=1024
FORMAT=pyaudio.paInt16
CHANNELS=2
RATE=44100
RECORD_SECONDS=2
WAVE_OUTPUT_FILENAME='userAudio.wav'

#variables for image processing
nrows=150
ncolumns=150
channels=3 #change to 1 if you want to use greyscale


CATEGORIES = ['aMajor','aMinor','bMinor','cMajor','dMajor',
		  'dMinor','eMajor','eMinor','fMajor','gMajor']

chords = ['aMajor','aMinor','bMinor','cMajor','dMajor',
		  'dMinor','eMajor','eMinor','fMajor','gMajor']
random.shuffle(chords)

def read_and_process_images(dir, list_of_images):
	'''
	returns X, an array of resizes imgs
	'''
	X = []
	for image in list_of_images:
		X.append(cv2.resize(cv2.imread(dir+image, cv2.IMREAD_COLOR), (nrows, ncolumns),
				 interpolation=cv2.INTER_CUBIC))
	return X

def getImageModelPred():
	'''
	returns the visual model's chord prediction and the probability associated with it
	'''
	#load model
	model = load_model("model.h5")
	#array only containing the image of the user's chord
	img = ["userPicture.jpg"]
	#resizes the user image 
	X= read_and_process_images("", img)
	#converts the resized image to a numpy array
	x = np.array(X)
	#test_datagen = ImageDataGenerator(rescale=1./255)
	#model's prediction of the chord the user played	
	pred= model.predict(x)
	#finds the index of the largest value in the predict array
	#this index corresponds to the chord at the same index in CATEGORIES
	index, value = max(enumerate(pred[0]), key=operator.itemgetter(1))
	print(CATEGORIES[index])
	return CATEGORIES[index], value

def getSoundModelPred():
	'''
	returns the audio model's chord prediction and the probability associated with it
	'''
	#load model
	model= load_model("audioModel.h5")
	#array only containing the frequency plot of the user's chord
	img=['userAudioFreq.jpg']
	#resizes the user image 
	X= read_and_process_images("", img)
	#converts the resized image to a numpy array
	x=np.array(X)
	#test_datagen=ImageDataGenerator(rescale=1./255)
	#model's prediction of the chord the user played
	pred= model.predict(x)
	#finds the index of the largest value in the predict array
	#this index corresponds to the chord at the same index in CATEGORIES
	index, value = max(enumerate(pred[0]), key=operator.itemgetter(1))
	print(CATEGORIES[index])
	return (CATEGORIES[index]), value


def audioToFreqPicture():
	'''
	converts input audio to frequency graph
	'''
	# read audio samples
	samplerate, data = wavfile.read('./userAudio.wav')
	print(str(samplerate))
	#number of samples in a the userAudio.wav audio clip
	samples= data.shape[0]
	#plot the first 2*samplerate samples
	plt.plot(data[:2*samplerate])
	#fourier transform of the audio data
	datafft=fft(data)
	#get the absolute value of real and complex component of the data
	fftabs=abs(datafft)
	freqs=fftfreq(samples,1/samplerate)
	#plot the frequency plot of the audio 
	#plt.plot(freqs,fftabs)
	
	#plt.xlim([10,samplerate/2])
	#plt.ylim([0,10000000])
	'''
	plt.xscale('log')
	'''
	plt.grid(True)	
	plt.xlabel('Frequency (hz)')
	#plot the frequency plot of the audio 
	plt.plot(freqs[:int(freqs.size/2)], fftabs[:int(freqs.size/2)])
	#save the plot as userAudioFreq.png
	plt.savefig('./'+'userAudioFreq' +'.png')
	#open and convert the image to rgb
	im=Image.open("./" + 'userAudioFreq' + '.png')
	rgb_im=im.convert('RGB')
	#save as jpeg
	rgb_im.save("./" + 'userAudioFreq' + '.jpg')
	#remove .png image
	os.remove("./" + 'userAudioFreq' + '.png')
	plt.close('all')	


def avgPreds():
	'''
	takes weighted average of two predictions and gives the most likely answer
	temporary method is a weighted average
	'''
	#predImg is the chord the model predicts, imgProb is the probability the model assigned to that chord
	predImg, imgProb = getImageModelPred()
	#predSound is the chord the model predicts, soundProb is the probability the model assigned to that chord
	predSound,soundProb = getSoundModelPred()	
	#if the predictions are the same, no calculation required	  
	if predImg==predSound:
		return predImg
	else:
		#expected value of the two probabilities
		expectedProb=0.6*(soundProb)+0.4*(imgProb)
		#if image model probability is closer to expected, then use the image model prediction
		if expectedProb-soundProb>expectedProb-imgProb:
			return predImg
		#if audio model probability is closer to expected, then use the audio model prediction
		else:
			return predSound

def run():
	#connects main computer camera
	camera = cv2.VideoCapture(0)
	#will be compared to time.time() at later intervals to see how many seconds have passed
	oldtime = time.time()
	#will be used to track each passing second by comparing to time.time()
	secondsDisplay = time.time()
	#number of seconds for timer
	seconds = 5
	#will determine chord the user will be tested on (based on index of chords array) 
	i=0
	#counter for how many chords the user will be tested on 
	iteration=0
	#keeps track of the user's score
	score=0
	
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
			iteration+=1
			print(str(iteration))
			i+=1
			i = i % len(chords)
			oldtime = time.time()
			secondsDisplay = time.time()
			seconds = 5
			
			#record audio at the time of capture
			p=pyaudio.PyAudio()
			stream = p.open(format=FORMAT,
                		channels=CHANNELS,
                		rate=RATE,
                		input=True,
                		frames_per_buffer=CHUNK)
			frames=[]
			for j in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    				data = stream.read(CHUNK)
    				frames.append(data)
			stream.stop_stream()
			stream.close()
			p.terminate()
			wf = wave.open('userAudio.wav', 'wb')
			wf.setnchannels(CHANNELS)
			wf.setsampwidth(p.get_sample_size(FORMAT))
			wf.setframerate(RATE)
			wf.writeframes(b''.join(frames))
			wf.close()
			#convert wav file to waveform picture
			audioToFreqPicture()
			
			#take picture of roi box after 5 seconds 
			cv2.imwrite('userPicture' + '.jpg', roi)
			
			#run neural networks to see if it matches the correct one
			answer = avgPreds()
			#if the chord the user plays as predicted by the neural network average matches the chord the software gave the user
			if (answer == chords[i-1]):
				print("Success")
				#increments the users score
				score+=1
				#show "success" on the screen
				cv2.putText(clone, "SUCCESS", (200,200),
					cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
				
			else:
				print("Fail")
				#show "fail" on the screen
				cv2.putText(clone, "FAIL", (200,200),
					cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
					
			#reset variables for the countdown for the next chord
			oldtime=time.time()
			seconds=5
			secondsDisplay=time.time()
			
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
		
		#after 20 chords, display your score
		if iteration >= 1:
			break
			
		#if you pressed 'q' then you quit
		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord('q'):
			break
			
	#connects main computer camera					
	camera2 = cv2.VideoCapture(0)
	# get the current frame
	(success2, frame2) = camera2.read()
	# resize the frame
	frame2 = imutils.resize(frame2, width=700)
	# flip the frame so that it is not mirror view
	frame2 = cv2.flip(frame2, 1)	
	clone2=frame2.copy()
	
	#display the score until the user quits
	while(True):
		cv2.putText(clone2, 'Score: ' + str(score) + '/20', (200,200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
		cv2.putText(clone2, 'Press "q" to exit.', (100,500), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
		cv2.imshow("Video Feed", clone2)
		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord('q'):
			break

run()
