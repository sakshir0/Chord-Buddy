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
channels=3 #change to 1 is you want to use greyscale


CATEGORIES = ['aMajor','aMinor','bMinor','cMajor','dMajor',
		  'dMinor','eMajor','eMinor','fMajor','eMinor',
		  'gMajor']

def read_and_process_images(dir, list_of_images):
	'''
	returns two arrays. X is array of resizes imgs. y is array of labels
	'''
	X = []
	y = []
	for image in list_of_images:
		X.append(cv2.resize(cv2.imread(dir+image, cv2.IMREAD_COLOR), (nrows, ncolumns),
				 interpolation=cv2.INTER_CUBIC))
		if 'aMajor' in image:
			y.append(0)
		elif 'aMinor' in image:
			y.append(1)
		elif 'bMinor' in image:
			y.append(2)
		elif 'cMajor' in image:
			y.append(3)
		elif 'dMajor' in image:
			y.append(4)
		elif 'dMinor' in image:
			y.append(5)
		elif 'eMajor' in image:
			y.append(6)
		elif 'eMinor' in image:
			y.append(7)
		elif 'fMajor' in image:
			y.append(8)
		elif 'gMajor' in image:
			y.append(9)
	return X, y

def getImageModelPred():
	#load model
	model = load_model("model.h5")
	img = ["userPicture.jpg"]
	X, y = read_and_process_images("", img)
	x = np.array(X)
	test_datagen = ImageDataGenerator(rescale=1./255)	
	pred= model.predict(x)
	index, value = max(enumerate(pred[0]), key=operator.itemgetter(1))
	print(CATEGORIES[index])
	return CATEGORIES[index],value

def getSoundModelPred():
	#load model
	model= load_model("audioModel.h5")
	img=['userAudioFreq.jpg']
	X, y = read_and_process_images("", img)
	x=np.array(X)
	test_datagen=ImageDataGenerator(rescale=1./255)
	pred= model.predict(x)
	index, value = max(enumerate(pred[0]), key=operator.itemgetter(1))
	print(CATEGORIES[index])
	return (CATEGORIES[index]), value

#converts input audio to frequency graph
def audioToFreqPicture():
	# read audio samples
	samplerate, data = wavfile.read('./userAudio.wav')
	samples= data.shape[0]
	plt.plot(data[:200])
	datafft=fft(data)
	fftabs=abs(datafft)
	freqs=fftfreq(samples,1/samplerate)
	plt.plot(freqs,fftabs)
	plt.xlim([10,samplerate/2])
	plt.ylim([0,10000000])
	plt.xscale('log')
	plt.grid(True)	
	plt.xlabel('Frequency (hz)')
	plt.plot(freqs[:int(freqs.size/2)], fftabs[:int(freqs.size/2)])
	
	plt.savefig('./'+'userAudioFreq' +'.png')
	im=Image.open("./" + 'userAudioFreq' + '.png')
	rgb_im=im.convert('RGB')
	rgb_im.save("./" + 'userAudioFreq' + '.jpg')
	os.remove("./" + 'userAudioFreq' + '.png')
	plt.close('all')	

#takes weighted average of two predictions and gives the most likely answer
#temporary method is a weighted average
def avgPreds():
	predImg, imgProb = getImageModelPred()
	predSound,soundProb = getSoundModelPred()		  
	if predImg==predSound:
		return predImg
	else:
		expectedProb=0.6*(soundProb)+0.4*(imgProb)
		#if image model probability is closer to expected, then use the image model prediction
		if expectedProb-soundProb>expectedProb-imgProb:
			return predImg
		else:
			return predSound
	

chords = ['aMajor','aMinor','bMinor','cMajor','dMajor',
		  'dMinor','eMajor','eMinor','fMajor','gMajor']
random.shuffle(chords)

def run():
	camera = cv2.VideoCapture(0)
	oldtime = time.time()
	secondsDisplay = time.time()
	seconds = 5
	i=0
	iteration=0
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
			if (answer == chords[i-1]):
				print("Success")
				cv2.putText(clone, "SUCCESS", (200,200),
							cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
				score+=1
			else:
				print("Fail")
				cv2.putText(clone, "FAIL", (200,200),
						cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
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
		if iteration >= 2:
			break

		#if you pressed 'q' then you quit
		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord('q'):
			break
			
	camera2 = cv2.VideoCapture(0)
	#would usually have a while true here, for actual user will have
	#for your purposes it is not there
	# get the current frame
	(success2, frame2) = camera2.read()
	# resize the frame
	frame2 = imutils.resize(frame2, width=700)
	# flip the frame so that it is not mirror view
	frame2 = cv2.flip(frame2, 1)	
	clone2=frame2.copy()
	
	while(True):
		cv2.putText(clone2, 'Score: ' + str(score) + '/20', (200,200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
		cv2.putText(clone2, 'Press "q" to exit.', (100,500), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
		cv2.imshow("Video Feed", clone2)
		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord('q'):
			break

run()
