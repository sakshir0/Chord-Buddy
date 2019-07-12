import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

#dimensions of the images (from tutorial)
img_width, img_height= 150, 150

top_model_weights_path="vgg19_weights.h5"
train_data_dir="train"
validation_data_dir="validation"
nb_train_samples=65
nb_validation_samples=9
epochs=50

#first the model wil be trained on the first batch_size, then the next batch_size, then the next batch_size
batch_size=1


def save_bottleneck_features():
	datagen=ImageDataGenerator(rescale=1./255)

	#build the VGG19 network
	model= applications.VGG19(include_top=False, weights= 'imagenet')
	
	#shuffle must be false unless you change the labels in train_top_model method
	generator=datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)
	
	#distinguish the bottle neck features (eliminates the top layer of VGG19 classifier)
	bottleneck_features_train=model.predict_generator(
		generator, 65)
	np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

	
	generator=datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None, 
		shuffle=False)
	#distinguish the bottle neck features of validation pictures
	bottleneck_features_validation=model.predict_generator(
		generator, 9)
	np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

#this will be specific to what we want the ouput to be
def train_top_model():
	train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
	train_labels = np.array([0] * 27 + [1] * 38)

	validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
	validation_labels = np.array([0] * (5) + [1] * (4))
	#our top model specifications
	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	
	model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
	
	model.fit(train_data, train_labels,
		epochs=epochs, 
		steps_per_epoch=nb_train_samples // batch_size,
		batch_size=None, 
		validation_steps=nb_validation_samples // batch_size,
		validation_data=(validation_data, validation_labels))
	model.save_weights(top_model_weights_path)

save_bottleneck_features()
train_top_model()








