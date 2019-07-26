import cv2
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

train_dir= 'chordPlots'
train_imgs=os.listdir(train_dir)
random.shuffle(train_imgs)

nrows=150
ncolumns=150
channels=3 #change to 1 is you want to use greyscale

def read_and_process_audio_images(dir, list_of_images):
	'''
	returns two arrays. X is array of resizes imgs. y is array of labels
	'''
	X = []
	y = []
	for image in list_of_images:
		X.append(cv2.resize(cv2.imread(dir+image, cv2.IMREAD_COLOR), (nrows, ncolumns),
				 interpolation=cv2.INTER_CUBIC))
		if 'am' in image:
			y.append(1)
		elif 'a' in image:
			y.append(0)
		elif 'bm' in image:
			y.append(2)
		elif 'c' in image:
			y.append(3)
		elif 'dm' in image:
			y.append(5)
		elif 'd' in image:
			y.append(4)
		elif 'em' in image:
			y.append(7)
		elif 'e' in image:
			y.append(6)
		elif 'f' in image:
			y.append(8)
		elif 'g' in image:
			y.append(9)
	return X, y

#gets arrays for training and validation data set
X, y = read_and_process_audio_images('chordPlots/', train_imgs)
X = np.array(X)
y = np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

ntrain = len(X_train)
nval = len(X_val)
#should be factor of 2
batch_size = 4

#using VGG neural net with added Flatten layer
#dropout layer which randomly drops some layers in order to prevent overfitting
#softmax activation since multiclass classification
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#get rid of water later
model.summary()
#compile model
#loss is categorical crossentropy because multiclass classification and because labels
#are mutually exclusive 
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Create augmentation configuration, which will help prevent overfitting
#imageDataGenerator decodes jpeg into rgb grid of pixels, 
#into floating point tensors, rescales pixels, and easily augments imgs
print("generating data")
train_datagen= ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)
print("data generation done")
#Create image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

#train the model. We train for 50 epochs with about 100 steps per epoch
history = model.fit_generator(train_generator,
							  steps_per_epoch=ntrain // batch_size,
							  epochs=64,
							  validation_steps = nval // batch_size,
							  validation_data=val_generator)

model.save_weights('model_weights_audio.h5')
model.save('audioModel.h5')
