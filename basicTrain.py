from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np


model = VGG19(weights='imagenet', include_top=False)

image = load_img('train/cMajor6.jpg', target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(x, axis=0)
image = preprocess_input(image)

features = model.predict(x)[0]
features_arr = np.char.mod('%f', features)
