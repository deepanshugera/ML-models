import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
# from keras.applications.resnet50 import conv_block, identity_block
from tensorflow.keras.layers import (Activation, BatchNormalization, Convolution2D, Dense, add,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Conv2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar100
from tensorflow.keras.utils import multi_gpu_model
import os
import time
bn_axis = 3
SHAPE = (32, 32, 3)


def unpickle():
    #data = {'labels': np.empty(), 'data': [], 'mean': np.empty()}
    my_dict={}
    files = os.listdir('/datastore/dataset/Imagenet32_train')
    for file in files:
        with open('/datastore/dataset/Imagenet32_train/'+file, 'rb') as fo:
            d = pickle.load(fo)
            
            if len(my_dict.keys()) == 0:
                my_dict = d
            else:
                print(len(my_dict['labels']))
                my_dict['data'] = np.concatenate((my_dict['data'], d['data']), axis = 0)
                my_dict['mean'] = np.concatenate((my_dict['mean'], d['mean']), axis = 0)
                my_dict['labels'] += d['labels']
                    
            #my_dict.update(pickle.load(fo))
        
    return my_dict

def unpickle_test(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

f = unpickle()
#f = unpickle_test('/mydata/dataset/Imagenet32_train/train_data_batch_1')
train_x = f['data']
train_x = train_x.reshape(train_x.shape[0], 3, 32, 32)
train_x = train_x.transpose(0, 2, 3, 1)
train_x = train_x/255.0

train_y = f['labels']
train_y = np.reshape(train_y, (len(train_y), 1))
train_y = to_categorical(train_y, num_classes=1001)

t = unpickle_test('/datastore/dataset/Imagenet32_val/val_data')
test = t['data']
test_x = test.reshape(test.shape[0], 3, 32, 32)
test_x = test_x.transpose(0, 2, 3, 1)
test_x = test_x/255.0

test_y = t['labels']
test_y = np.reshape(test_y, (len(test_y), 1))
test_y = to_categorical(test_y, num_classes=1001)

# ------------------------ TESTING WITH CIFAR --------------------------#
#(train_x, train_y), (test_x, test_y) = cifar100.load_data()
#train_y = to_categorical(train_y, num_classes=100)
#train_x = train_x.astype(float)/255.0
#test_x = test_x.astype(float)/255.0
# ----------------------------------------------------------------------#

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization( axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def build_model(seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    x = ZeroPadding2D((3, 3))(input_layer)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    """print(x)"""
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    

    """ x = Flatten()(x)"""
    x = Dense(1001, activation='softmax', name='fc10')(x)

    model = Model(input_layer, x)

    return model


# fit
model = build_model()
# model = Parallelizer().transform(model)
# Quantization aware training
sess = tf.keras.backend.get_session()
tf.contrib.quantize.create_training_graph(sess.graph, quant_delay=0)
sess.run(tf.global_variables_initializer())

#model = multi_gpu_model(model, gpus=4)
model.compile(RMSprop(lr=1e-4), 'categorical_crossentropy', ['accuracy'])
# batch_size = real_batch_size * n_GPUs
# model.fit(train_x, train_y, batch_size=64*2, nb_epoch=20)
model.fit(train_x, train_y, batch_size=256, nb_epoch=1)
# model.save('digit_recognizer_model.h5')

# predict
pred_y = model.predict(test_x).argmax(1)
pd.DataFrame({
    'ImageId': range(1, len(pred_y)+1),
    'Label': pred_y
}).to_csv('test_y.csv', index=False)
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(test_x, test_y)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)
time.sleep(60)
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
converter.default_ranges_stats = (0,6)
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 255.)}  # mean, std_dev
tflite_model = converter.convert()
open("quantized_model.tflite", "wb").write(tflite_model)
