from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# load  model
model = load_model("model.h5")

#load image from file
#img = cv2.imread('validation/cMajor/cMajor1.jpg')

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

