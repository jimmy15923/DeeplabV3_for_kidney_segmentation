import glob
import pandas 
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import sys
# sys.path.append('keras-deeplab-v3-plus/')
# from deeplab_v3_plus.model import *

import os
import random
import time
import skimage
from skimage import io, img_as_bool
from skimage.transform import resize


# config
crop_size = 512

def get_unet():
  
    inputs = tf.keras.layers.Input((crop_size, crop_size, 3))
    
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='SAME')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='SAME')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='SAME')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='SAME')(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME')(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(conv7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv8)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(conv8), conv1], axis=3)
    
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(up9)
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv9)

    conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    return model

def read_data_and_split(split_seed, train_ratio, is_normalize=True, is_resize=crop_size):
    """read data into np array, normalize it and train test split
    
    split_seed: set seed for same train test split
    train_ratio: ratio of training set. range from 0 to 1
    is_normalize: True for normalizr to -1 to 1
    
    return: x_train, x_test, y_train, y_test
    """
    
    idx = next(os.walk('dataset/'))[1]
    # remove two file with different size between image & mask
    idx.remove("S2016-30816_9_0")
    idx.remove("S2016-30816_9_1")
    
    # set seed
    random.seed(split_seed)
    random.shuffle(idx)
    
    train_idx, test_idx = idx[:int(len(idx)*train_ratio)], idx[int(len(idx)*train_ratio):]

    x_train = np.array([skimage.io.imread('dataset/{}/image/{}_slide.jpg'.format(x, x))\
                    for x in train_idx], dtype="float32")
    x_test = np.array([skimage.io.imread('dataset/{}/image/{}_slide.jpg'.format(x, x))\
                       for x in test_idx], dtype="float32")
    
    if is_normalize:
        x_train = (x_train / 127.5) - 1
        x_test = (x_test / 127.5) - 1
        
    y_train = np.array([skimage.io.imread('dataset/{}/mask/{}_mask.jpg'.format(x, x))\
                    for x in train_idx])
    
    y_test = np.array([skimage.io.imread('dataset/{}/mask/{}_mask.jpg'.format(x, x))\
                        for x in test_idx])
    
    y_train = np.expand_dims(y_train, 3)
    y_test = np.expand_dims(y_test, 3)
    
    def cv2_resize(array):
        return np.array([resize(x, (crop_size, crop_size)) for x in array])
    
    if is_resize:
        x_train, x_test, y_train, y_test = cv2_resize(x_train), cv2_resize(x_test), cv2_resize(y_train), cv2_resize(y_test)
    
    return x_train, x_test, y_train, y_test

def data_gen(x_train, y_train, bz, augmentation=None):
    i = 0
    from sklearn.utils import shuffle
    while True:
        if i == len(y_train) // bz:
            i = 0
            x_train, y_train = shuffle(x_train, y_train)
            
        x_, y_ = x_train[i*bz:(i+1)*bz], y_train[i*bz:(i+1)*bz]
#         img_idx = np.random.choice(range(len(y_train)), bz, replace=False)
        i += 1
        yield x_, y_
        
def IOU_cal(y_true, y_pred):
    iou_res = []
    for i in range(len(y_true)):
        y_true_flat = y_true[i].ravel()
        y_pred_flat = (y_pred[i][...,1].ravel() > 0.5) * 1
        intersection = np.sum(y_true_flat * y_pred_flat)
        union = np.sum((y_true_flat+y_pred_flat) - (y_true_flat*y_pred_flat))
        iou = intersection / union
        iou_res.append(iou)
    return iou_res        
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


    
x_train, x_test, y_train, y_test = read_data_and_split(split_seed=7, train_ratio=0.8)


# y_train_inv = np.where(y_train, 0, 1)
# y_train_ = np.zeros(shape=(len(y_train), crop_size, crop_size, 2))
# y_train_[:,:,:,0] = y_train
# y_train_[:,:,:,1] = y_train_inv

# y_test_inv = np.where(y_test, 0, 1)
# y_test_ = np.zeros(shape=(len(y_test), crop_size, crop_size, 2))
# y_test_[:,:,:,0] = y_test
# y_test_[:,:,:,1] = y_test_inv

print("Data shape:")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#     model = Deeplabv3(input_shape=(crop_size, crop_size, 3), classes=2, OS=8)
#     logits = model.output
#     output = tf.keras.layers.Activation("softmax")(logits)
#     model = tf.keras.models.Model(model.input, output)

model = get_unet()

def dice_coef_loss(y_true, y_pred, smooth=1):
    def dice_coef_fix(y_true, y_pred):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis = -1)
        iou = (2. * intersection + smooth) / (tf.keras.backend.sum(tf.keras.backend.square(y_true), -1) +\
                                              tf.keras.backend.sum(tf.keras.backend.square(y_pred),-1) + smooth)
        return iou
    loss = 1 - dice_coef_fix(y_true, y_pred)
    return loss

#     model_gpu = tf.keras.utils.multi_gpu_model(model, gpus=args.num_gpus)

model.compile(optimizer="Adam",
              loss=dice_coef_loss)

early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, verbose=1)
check = tf.keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                        filepath="test_resize.h5",
                                        verbose=1, save_best_only=True)

reduce = tf.keras.callbacks.ReduceLROnPlateau(patience=3)

t = time.time()
model.fit_generator(data_gen(x_train, y_train, 12),
                    steps_per_epoch=60,
                    epochs=30, 
                    validation_data=(x_test, y_test),
                    callbacks=[early, check, reduce]
                   )
model = tf.keras.models.load_model("test_resize.h5")
## inference
_, x_test, _, y_test = read_data_and_split(split_seed=7, train_ratio=0.8, is_resize=1000)

y_pred = model.predict(x_test)

res_iou = IOU_cal(y_test, y_pred)
print("TESTING IOU: ",np.mean(res_iou))
print("-"*10)
print("Elapse time:", time.time() - t)