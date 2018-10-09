import glob
import pandas 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.cross_validation import train_test_split
import random
from keras import backend as K
import keras
from keras.utils import to_categorical
from keras.utils.training_utils import multi_gpu_model

import os
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())

import sys
sys.path.append('keras-deeplab-v3-plus/')
from deeplab_v3_plus.model import *

import cv2
import numpy as np
import os
import random
from skimage import io
from skimage.transform import resize
from skimage import img_as_bool

def read_data_and_split(split_seed, train_ratio, is_normalize=True):
    """read data into np array, normalize it and train test split
    split_seed: set seed for same train test split
    train_ratio: ratio of training set. range from 0 to 1
    is_normalize: True for normalizr to -1 to 1
    
    return np array with x_train, x_test, y_train, y_test
    """
    
    idx = next(os.walk('/data/jimmy15923/cg_kidney_seg/train'))[1]
    # remove two file with different size between image & mask
    idx.remove("S2016-30816_9_0")
    idx.remove("S2016-30816_9_1")
    
    # set seed
    random.seed(split_seed)
    random.shuffle(idx)
    
    train_idx, test_idx = idx[:int(len(idx)*train_ratio)], idx[int(len(idx)*train_ratio):]

    x_train = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/image/{}_slide.jpg'.format(x, x))[...,::-1]\
                    for x in train_idx], dtype="float32")
    x_test = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/image/{}_slide.jpg'.format(x, x))[...,::-1]\
                       for x in test_idx], dtype="float32")
    
    if is_normalize:
        x_train = (x_train / 127.5) - 1
        x_test = (x_test / 127.5) - 1
        
    y_train = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/mask/{}_mask.jpg'.format(x, x))[..., 0]\
                    for x in train_idx])
    
    y_test = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/mask/{}_mask.jpg'.format(x, x))[..., 0]\
                        for x in test_idx])
    
    y_train = img_as_bool(y_train)
    y_test = img_as_bool(y_test)
    
    return x_train, x_test, y_train, y_test

def cv2_resize(array):
    return np.array([resize(x, (500,500)) for x in array])

x_train, x_test, y_train, y_test = read_data_and_split(split_seed=7, train_ratio=0.8, is_normalize=True)

x_train = cv2_resize(x_train)
x_test = cv2_resize(x_test)
y_train = cv2_resize(y_train)
y_test = cv2_resize(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train_inv = np.where(y_train, 0, 1)
y_train_ = np.zeros(shape=(len(y_train), 500,500, 2))
y_train_[:,:,:,0] = y_train
y_train_[:,:,:,1] = y_train_inv

y_test_inv = np.where(y_test, 0, 1)
y_test_ = np.zeros(shape=(len(y_test), 500,500, 2))
y_test_[:,:,:,0] = y_test
y_test_[:,:,:,1] = y_test_inv

# def data_gen(x_train, y_train, bz, augmentation=None):
#     i = 0
#     from sklearn.utils import shuffle
#     while True:
#         if i == len(y_train):
#             i = 0
#             x_train, y_train = shuffle(x_train, y_train)
            
#         x_, y_ = x_train[i*bz:(i+1)*bz], y_train[i*bz:(i+1)*bz]
        
#         i+=1
#         yield x_, y_
        

def data_gen(x_train, y_train, bz, augmentation=None):
    i = 0
    from sklearn.utils import shuffle
    while True:
#         if i == len(y_train):
#             i = 0
#             x_train, y_train = shuffle(x_train, y_train)
            
#         x_, y_ = x_train[i*bz:(i+1)*bz], y_train[i*bz:(i+1)*bz]
        img_idx = np.random.choice(range(len(y_train)), bz, replace=False)
        

        yield x_train[img_idx], y_train[img_idx]
        
        
# def val_gen(x_test, y_test, crop_size=500, stride=500):
#     i = 0
#     while True:
#         x = []
#         y = []
#         for x_start in range(0, crop_size+1, stride):
#             for y_start in range(0, crop_size+1, stride):
#                 x_crop = x_test[i][x_start:(x_start+crop_size), y_start:(y_start+crop_size), :]
#                 y_crop = y_test[i][x_start:(x_start+crop_size), y_start:(y_start+crop_size), :]
#                 x.append(x_crop)
#                 y.append(y_crop)
#         i+=1
#         yield np.array(x), np.array(y)
#         if i == len(y_test):
#             i=0

crop_size = 500        
import tensorflow as tf
with tf.device('/cpu:0'):
    model = Deeplabv3(input_shape=(crop_size, crop_size, 3), classes=2, OS=8)
    logits = model.output
    output = keras.layers.Activation("softmax")(logits)
    model = Model(model.input, output)

def dice_coef_loss(y_true, y_pred, smooth = 1):
    def dice_coef_fix(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
        iou = (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred),-1) + smooth)
        return iou
    loss = 1 - dice_coef_fix(y_true, y_pred)
    return loss

model_gpu = multi_gpu_model(model, gpus=2)

model_gpu.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True),
              loss=dice_coef_loss)

early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, verbose=1)
check = keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                        filepath="/data/jimmy15923/cg_kidney_seg/test_resize_dice.h5",
                                        verbose=1, save_best_only=True, save_weights_only=True)

reduce = keras.callbacks.ReduceLROnPlateau(patience=3)


model_gpu.fit_generator(data_gen(x_train, y_train_, 12),
                    steps_per_epoch=200,
                    epochs=1000, 
                    validation_data=(x_test, y_test_),
                    callbacks=[early, check, reduce]
                   )