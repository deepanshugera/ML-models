import pickle
import numpy as np
import pandas as pd
from keras import backend as K
# from keras.applications.resnet50 import conv_block, identity_block
from keras.layers import (Activation, BatchNormalization, Convolution2D, Dense,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Conv2D)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import os

bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1
SHAPE = (32, 32, 3)

def unpickle():
    #data = {'labels': np.empty(), 'data': [], 'mean': np.empty()}
    my_dict={}
    files = os.listdir('./train_data')
    for file in files:
        with open('./train_data/'+file, 'rb') as fo:
            d = pickle.load(fo)
            
            if len(my_dict.keys()) == 0:
                my_dict = d
            else:
                print(len(my_dict['labels']))
                my_dict['data'] = np.concatenate((my_dict['data'], d['data']), axis = 0)
                my_dict['mean'] = np.concatenate((my_dict['mean'], d['mean']), axis = 0)
                my_dict['labels'] += d['labels']
                    
    return my_dict



def unpickle_test(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


f = unpickle()
train_x = f['data']
train_x = train_x.reshape(train_x.shape[0], 3, 32, 32)
train_x = train_x.transpose(0, 2, 3, 1)
train_x = train_x.astype(float)/255.0

train_y = f['labels']
train_y = np.reshape(train_y, (len(train_y), 1))
train_y = to_categorical(train_y, num_classes=1001)

t = unpickle_test('/val_data')
test = t['data']
test_x = test.reshape(test.shape[0], 3, 32, 32)
test_x = test_x.transpose(0, 2, 3, 1)
test_x = test_x.astype(float)/255.0


def build_model(seed=None):
    
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(input_layer)
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', dim_ordering = 'tf')(x)
    
    
    
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', dim_ordering = 'tf')(x)
    
    
    
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', dim_ordering = 'tf')(x)
    
    
    
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', dim_ordering = 'tf')(x)
    
    
    
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', dim_ordering = 'tf')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1001, activation='softmax', name='fc10')(x)
    model = Model(input_layer, x)

    return model

# fit
model = build_model()
# model = Parallelizer().transform(model)
model.compile(RMSprop(lr=1e-4), 'categorical_crossentropy', ['accuracy'])
# batch_size = real_batch_size * n_GPUs
# model.fit(train_x, train_y, batch_size=64*2, nb_epoch=20)
model.fit(train_x, train_y, batch_size=64, nb_epoch=50)
# model.save('digit_recognizer_model.h5')

# predict
pred_y = model.predict(test_x).argmax(1)
pd.DataFrame({
    'ImageId': range(1, len(pred_y)+1),
    'Label': pred_y
}).to_csv('test_y.csv', index=False)