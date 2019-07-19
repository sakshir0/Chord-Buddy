import os
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from os import walk
from PIL import Image

DATADIR="./recordings"

#creates new chordPlots folder if it doesn't exist already
if not os.path.exists("chordPlots"):
	os.makedirs("chordPlots")

chord_wavs = []

#goes through each file in the directory
#chord_wavs will contain the list of all the names of the audio files after loop finishes
for (_,_,filenames) in walk(DATADIR):
	chord_wavs.extend(filenames)

for chord_wav in chord_wavs:
	# read audio samples
	input_data = read(DATADIR + "/" + chord_wav)
	audio = input_data[1]

	# plot the first samples
	plt.plot(audio)

	# label the axes
	plt.ylabel("Amplitude")
	plt.xlabel("Time")

	# set the title
	plt.title(chord_wav + " Wav")
	# display the plot
	plt.savefig("chordPlots/" + chord_wav.split('.')[0] + '.png')
	
	#convert the image to jpeg
	im=Image.open("chordPlots/" + chord_wav.split('.')[0] + '.png')
	rgb_im=im.convert('RGB')
	rgb_im.save("chordPlots/" + chord_wav.split('.')[0] + '.jpg')
	os.remove("chordPlots/" + chord_wav.split('.')[0] + '.png')
	
	# plt.show()
	plt.close('all')

