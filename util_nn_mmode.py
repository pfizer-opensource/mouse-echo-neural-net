# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:38:50 2019

@author: DUANC01
"""

from __future__ import print_function

from skimage.transform import resize
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.exposure import rescale_intensity
from keras.callbacks import History
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation

# Some parameters
K.set_image_data_format('channels_last')  # TF dimension ordering in this code 
img_rows = 256
img_cols = 256

# metric and loss functions
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# def generalized_dice_coef(y_true, y_pred):
#     # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
#     Ncl = y_pred.shape[-1]
#     w = np.zeros((Ncl,))
#     for l in range(0, Ncl):
#         w[l] = np.sum(np.asarray(y_true[:, :, :, l] == 1, np.int8))
#     w = 1 / (w ** 2 + 0.00001)
# 
#     # Compute gen dice coef:
#     numerator = y_true * y_pred
#     numerator = K.sum(numerator, (0, 1, 2))
#     numerator = K.sum(numerator)
# 
#     denominator = y_true + y_pred
#     denominator = w * K.sum(denominator, (0, 1, 2))
#     denominator = K.sum(denominator)
# 
#     gen_dice_coef = numerator / denominator
# 
#     return 1 - 2 * gen_dice_coef


def dice_coef_multilabel(y_true, y_pred):
    # n_class = y_true.shape[-1]
    dice = 0
    for index in range(4):
        dice += dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice/4.0


def dice_coef_multilabel_loss(y_true, y_pred):
    return -dice_coef_multilabel(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# def dice_coef_square(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)

# def dice_coef_loss_square(y_true, y_pred):
#     return -dice_coef_square(y_true, y_pred)


# Vanilla U-Net
def get_unet(numLabels=4):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # multiclass
    conv10 = Conv2D(numLabels, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel, 'accuracy'])

    return model
