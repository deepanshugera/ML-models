#!/usr/bin/env python
# coding: utf-8
import pickle
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import (Activation, BatchNormalization, Convolution2D, Dense,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Conv2D)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import os

bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1
SHAPE = (32, 32, 3)


def unpickle():
    # data = {'labels': np.empty(), 'data': [], 'mean': np.empty()}
    my_dict = {}
    files = os.listdir('../dataset/Imagenet32_train')
    for file in files:
        with open('../dataset/Imagenet32_train/' + file, 'rb') as fo:
            d = pickle.load(fo)

            if len(my_dict.keys()) == 0:
                my_dict = d
            else:
                print(len(my_dict['labels']))
                my_dict['data'] = np.concatenate((my_dict['data'], d['data']), axis=0)
                my_dict['mean'] = np.concatenate((my_dict['mean'], d['mean']), axis=0)
                my_dict['labels'] += d['labels']

            # my_dict.update(pickle.load(fo))

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

t = unpickle_test('../dataset/Imagenet32_val/val_data')
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
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', dim_ordering='tf')(x)

    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', dim_ordering='tf')(x)

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
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv4')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', dim_ordering='tf')(x)

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
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', dim_ordering='tf')(x)

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
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', dim_ordering='tf')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1001, activation='softmax', name='predictions')(x)

    model = Model(input_layer, x)

    return model

# fit
model = build_model()


# model = Parallelizer().transform(model)
model.compile(RMSprop(lr=1e-4), 'categorical_crossentropy', ['accuracy'])

model.fit(train_x, train_y, batch_size=256, nb_epoch=50)

# predict
pred_y = model.predict(test_x).argmax(1)
pd.DataFrame({
    'ImageId': range(1, len(pred_y)+1),
    'Label': pred_y
}).to_csv('test_y.csv', index=False)

