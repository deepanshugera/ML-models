#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
#from keras.applications.resnet import ResNet152, preprocess_input, decode_predictions
from keras_applications.resnet import ResNet152, preprocess_input, decode_predictions
import keras
from keras.preprocessing import image
import numpy as np
import os
from datetime import datetime

imgs = []
basepath = 'input/Inference/images'
img_paths = os.listdir(basepath)

for path in img_paths:
    img_path = basepath + '/' + path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    imgs.append(x)

x = np.stack(imgs)
x = preprocess_input(x)
model = ResNet50(weights='imagenet')
start = datetime.now()
preds = model.predict(x)
end = datetime.now()
decoded = decode_predictions(preds, top=1)

print('Inference start time:', str(start))
print('Inference end time:', str(end))
print('Inference delta time:', str(end-start))
for pred in decoded:
    print('Predicted:', pred)
