import glob
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.cross_validation import train_test_split
import random
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import os
import tensorflow as tf
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())

import sys
sys.path.append('keras-deeplab-v3-plus/')
sys.path.append("keras_retinanet")
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
    df = pd.read_csv("/data/jimmy15923/cg_kidney_seg/cgmh_data_list.csv")
    df = df[df.n_mask_pixel > 1000].copy()
    
    idx = df.uid.tolist()
    
#     idx = next(os.walk('/data/jimmy15923/cg_kidney_seg/train'))[1]
#     # remove two file with different size between image & mask
#     idx.remove("S2016-30816_9_0")
#     idx.remove("S2016-30816_9_1")
    
    # set seed
    random.seed(split_seed)
    random.shuffle(idx)
    
    train_idx, test_idx = idx[:int(len(idx)*train_ratio)], idx[int(len(idx)*train_ratio):]

    x_train = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/image/{}_slide.jpg'.format(x, x))[...,::-1]\
                    for x in train_idx], dtype="uint8")
    x_test = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/image/{}_slide.jpg'.format(x, x))[...,::-1]\
                       for x in test_idx], dtype="uint8")
    
    if is_normalize:
        x_train = (x_train / 127.5) - 1
        x_test = (x_test / 127.5) - 1
        
    y_train = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/mask/{}_mask.jpg'.format(x, x))[..., 0]\
                    for x in train_idx])
    
    y_test = np.array([cv2.imread('/data/jimmy15923/cg_kidney_seg/train/{}/mask/{}_mask.jpg'.format(x, x))[..., 0]\
                        for x in test_idx])
    
    y_train = y_train.astype(np.bool)
    y_test = y_test.astype(np.bool)
    
    return x_train, x_test, y_train, y_test

def cv2_resize(array, size=500, is_bool=False):
    if is_bool:
        return np.array([img_as_bool(resize(x, (size, size))) for x in array])
    return np.array([resize(x, (size,size), preserve_range=True).astype("uint8") for x in array])

size=500
binarize=False

x_train, x_test, y_train, y_test = read_data_and_split(split_seed=7, train_ratio=0.9, is_normalize=False)

x_train = cv2_resize(x_train, size)
x_test = cv2_resize(x_test, size)
y_train = cv2_resize(y_train, size, is_bool=True)
y_test = cv2_resize(y_test, size, is_bool=True)
x_test = (x_test / 127.5) - 1

if binarize:
    y_train = np.expand_dims(y_train, 3)
    y_test = np.expand_dims(y_test, 3)
else:
    y_train = np.stack((~y_train, y_train), axis=3)
    y_test = np.stack((~y_test, y_test), axis=3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras import backend as K
from keras_retinanet import backend
'''
Compatible with tensorflow backend
'''
def dice_coef_loss(y_true, y_pred, smooth = 1):
    def dice_coef_fix(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
        iou = (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred),-1) + smooth)
        return iou
    loss = 1 - dice_coef_fix(y_true, y_pred)
    return loss

def focal_loss(gamma=2, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

import tensorflow as tf
with tf.device('/cpu:0'):
    model = Deeplabv3(weights=None, input_shape=(size, size, 3), classes=2, OS=8, alpha=1.5)
    logits = model.output
    output = keras.layers.Activation("softmax")(logits)
    model = Model(model.input, output)
    
model_gpu = multi_gpu_model(model, gpus=3)

def data_gen(x_train, y_train, bz, augmentation=None):
    from sklearn.utils import shuffle
    x_train, y_train = shuffle(x_train, y_train)
    steps = len(x_train) // bz
    n = 0
    while True:
        if n == steps:
            n=0
            x_train, y_train = shuffle(x_train, y_train)
            
        x, y = x_train[n*bz:(n+1)*bz], y_train[n*bz:(n+1)*bz]
        n+=1

        if augmentation:
            import imgaug
            # Augmentors that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]
            
            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            x = det.augment_images(x)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            y = det.augment_images(y.astype(np.uint8),
                                     hooks=imgaug.HooksImages(activator=hook))
            
            x = (x / 127.5) - 1

        yield x, y

from imgaug import augmenters as iaa
model_gpu.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=dice_coef_loss)

early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1)
check = keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                        filepath="/data/jimmy15923/cg_kidney_seg/deepx_resize_aug.h5",
                                        verbose=1, save_best_only=True)

reduce = keras.callbacks.ReduceLROnPlateau(patience=3)

augmentation = iaa.SomeOf((0, 4), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        iaa.AddToHueAndSaturation()
    ])


from keras.callbacks import Callback

class MutliGPU_ModelCheckpoint(Callback):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', period=1):
        super(MutliGPU_ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.model_to_save = model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.model_to_save.save(filepath, overwrite=True)

                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))

                    self.model_to_save.save(filepath, overwrite=True)

gpu_check = MutliGPU_ModelCheckpoint(model, "/data/jimmy15923/cg_kidney_seg/deeplab_alpha_dice.h5", verbose=1, save_best_only=True)
model_gpu.fit_generator(data_gen(x_train, y_train, 6, augmentation),
                    steps_per_epoch=100,
                    epochs=10000, 
                    validation_data=(x_test, y_test),
                    callbacks=[early, gpu_check, reduce]
                   )