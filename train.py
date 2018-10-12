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
batch_size = 8

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2)) (c1)
    p1 = tf.keras.layers.Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2)) (c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2)) (c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = tf.keras.layers.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = tf.keras.layers.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = tf.keras.layers.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = tf.keras.layers.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    u9 = tf.keras.layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = tf.keras.models.Model(inputs=[input_img], outputs=[outputs])
    return model

def image_pad(image):
    h, w, _ = image.shape
    # Height
    if h % 64 > 0:
        max_h = h - (h % 64) + 64
        top_pad = (max_h - h) // 2
        bottom_pad = max_h - h - top_pad
    else:
        top_pad = bottom_pad = 0
    # Width
    if w % 64 > 0:
        max_w = w - (w % 64) + 64
        left_pad = (max_w - w) // 2
        right_pad = max_w - w - left_pad
    else:
        left_pad = right_pad = 0
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image

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
    
    def arr_resize(array):
        return np.array([resize(x, (crop_size, crop_size)) for x in array])
    
    def arr_pad(array):
        return np.array([image_pad(x) for x in array])

    if is_resize:
        if is_resize <= 1000:
            x_train, x_test, y_train, y_test = arr_resize(x_train), arr_resize(x_test), arr_resize(y_train), arr_resize(y_test)
        else:       
            x_train, x_test, y_train, y_test = arr_pad(x_train), arr_pad(x_test), arr_pad(y_train), arr_pad(y_test)
    
    return x_train, x_test, y_train, y_test

def data_gen(x_train, y_train, bz, augmentation=None):
    i = 0
    from sklearn.utils import shuffle
    while True:
#         if i == len(y_train) // bz:
#             i = 0
#             x_train, y_train = shuffle(x_train, y_train)
            
#         x_, y_ = x_train[i*bz:(i+1)*bz], y_train[i*bz:(i+1)*bz]
        img_idx = np.random.choice(range(len(y_train)), bz, replace=False)
        i += 1
        yield x_train[img_idx], y_train[img_idx]
        
def IOU_cal(y_true, y_pred):
    iou_res = []
    for i in range(len(y_true)):
        y_true_flat = y_true[i].ravel()
        y_pred_flat = (y_pred[i][...,0].ravel() > 0.5) * 1
        intersection = np.sum(y_true_flat * y_pred_flat)
        union = np.sum((y_true_flat+y_pred_flat) - (y_true_flat*y_pred_flat))
        iou = intersection / union
        iou_res.append(iou)
    return iou_res     


x_train, x_test, y_train, y_test = read_data_and_split(split_seed=7, train_ratio=0.8, is_resize=crop_size)

print("Data shape:")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

def dice_metric(y_true, y_pred, smooth = 1.):   
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)    
    return (2.*intersection + smooth)  / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_metric(y_true, y_pred)

input_img = tf.keras.layers.Input((crop_size, crop_size, 3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, verbose=1)
check = tf.keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                        filepath="test_resize.h5",
                                        verbose=1, save_best_only=True)
reduce = tf.keras.callbacks.ReduceLROnPlateau(patience=3)

t = time.time()
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=10, 
                    validation_data=(x_test, y_test),
                    callbacks=[early, check, reduce]
                   )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("learning_curve.png", dpi=500)

model = tf.keras.models.load_model("test_resize.h5", custom_objects = {'mean_iou': mean_iou,
                                                                       'dice_metric': dice_metric})
## inference
_, x_test, _, y_test = read_data_and_split(split_seed=7, train_ratio=0.8, is_resize=crop_size)

y_pred = model.predict(x_test, batch_size=batch_size)

res_iou = IOU_cal(y_test, y_pred)
print("TESTING IOU: ",np.mean(res_iou))
print("-"*10)
print("Elapse time:", time.time() - t)