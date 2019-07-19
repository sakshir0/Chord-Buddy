import os
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, 
outputs=base_model.get_layer('flatten').output)

def get_features(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	flatten = model.predict(x)
	return list(flatten[0])

X= []
y= []

chord_plots=[]
counter=0
for img in os.listdir('./chordPlots'):
	X.append(get_features('chordPlots/'+img))
	if 'am' in img:
		y.append(1)
	elif 'a' in img:
		y.append(0)
	elif 'bm' in img:
		y.append(2)
	elif 'c' in img:
		y.append(3)
	elif 'dm' in img:
		y.append(5)
	elif 'd' in img:
		y.append(4)
	elif 'em' in img:
		y.append(7)
	elif 'e' in img:
		y.append(6)
	elif 'f' in img:
		y.append(8)
	elif 'g' in img:
		y.append(9)
	
	counter+=1
	
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42, stratify=y)

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

# get the accuracy
print (accuracy_score(y_test, predicted))