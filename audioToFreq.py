import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from os import walk
from scipy.fftpack import fft, fftfreq
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
	samplerate, data = wavfile.read(DATADIR + "/" + chord_wav)
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
	
	plt.savefig('chordPlots/'+chord_wav.split('.')[0] +'.png')
	im=Image.open("chordPlots/" + chord_wav.split('.')[0] + '.png')
	rgb_im=im.convert('RGB')
	rgb_im.save("chordPlots/" + chord_wav.split('.')[0] + '.jpg')
	os.remove("chordPlots/" + chord_wav.split('.')[0] + '.png')
	plt.close('all')