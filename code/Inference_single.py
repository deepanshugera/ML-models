from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
#from keras.applications.resnet import ResNet152, preprocess_input, decode_predictions
import keras
from keras.preprocessing import image
import numpy as np
from datetime import datetime
import os

basepath = 'input/Inference/images'
img_paths = os.listdir(basepath)
#preds = []
model = ResNet50(weights='imagenet')
delta = 0

for path in img_paths:
    img_path = basepath + '/' + path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	start = datetime.now()
	pred = model.predict(x)
	end = datetime.now()
	delta += (end-start).total_seconds()

print('Inference delta time:', str(delta))
