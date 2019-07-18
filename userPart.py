from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

#load model
model = load_model("model.h5")

img = ['validation/cMajor/cMajor1.jpg']
X, y = read_and_process_image(img)
x = np.array(X)
test_datagen = ImageDataGenerator(rescale=1./255)
for batch in test_datagen.flow(x, batch_size=1):
	pred = model.predict(batch)
#test_datagen = ImageDataGenerator(rescale=1./255)
#for batch in test_datagen
print(X[0])
print(y)

'''
def load_image(image_path, grayscale=False, target_size=None):
    color_mode = 'grayscale'
    if grayscale == False:
        color_mode = 'rgb'
    else:
        grayscale = False
    pil_image = load_img(image_path, grayscale, color_mode, target_size)
    return img_to_array(pil_image)

img = load_image('validation/cMajor/cMajor1.jpg',False,(150, 150))

x = img_to_array(img)
prediction = model.predict(x.reshape((1,3,150, 150)),batch_size=1, verbose=0)
print(prediction)
'''

