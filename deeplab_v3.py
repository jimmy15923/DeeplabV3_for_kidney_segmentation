import glob
import pandas as pd
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
from keras.models import load_model

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())

import os
import tensorflow as tf
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

import tensorflow as tf

model = Deeplabv3(input_shape=(size, size, 3), classes=2, OS=8)
logits = model.output
output = keras.layers.Activation("softmax")(logits)
model = Model(model.input, output)


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
model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True),
              loss='categorical_crossentropy')

early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, verbose=1)
check = keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                        filepath="/data/jimmy15923/cg_kidney_seg/deeplab_resize_aug.h5",
                                        verbose=1, save_best_only=True)

reduce = keras.callbacks.ReduceLROnPlateau(patience=3)

augmentation = iaa.SomeOf((0, 4), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        iaa.AddToHueAndSaturation()
    ])

# gpu_check = MutliGPU_ModelCheckpoint(model, "/data/jimmy15923/deepx_resize_aug.h5")
history = model.fit_generator(data_gen(x_train, y_train, 8, augmentation),
                    steps_per_epoch=100,
                    epochs=1000, 
                    validation_data=(x_test, y_test),
                    callbacks=[early, check, reduce]
                   )

res = pd.DataFrame(history.history)
best_acc = res["val_loss"].min()
print("BEST Loss =", best_acc)
res.to_csv("/data/jimmy15923/cg_kidney_seg/performance.csv")
test_df.to_csv("/data/jimmy15923/cg_kidney_seg/prediction.csv")