# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:33:27 2019

@author: Zhenwei Feng, Shuai Jia
"""

import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


initial_filters = 16

def initial_block(inputs):
    i1 = Conv2D(filters = initial_filters, kernel_size = 7, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(inputs)
    i1 = BatchNormalization()(i1)
    i1 = Dropout(0.2)(i1)
    i1 = Activation('relu')(i1)
    i1 = MaxPooling2D(strides = 2)(i1)
    #model_init = Model(inputs = inputs, outputs = i1)
    return i1


# =============================================================================
# def encoder_block(inputs, filters = 64):
#     residual = inputs
#     #first conv
#     x1 = Conv2D(filters = filters, kernel_size = 3, strides = 2, padding = 'same',kernel_initializer = 'he_normal')(inputs)
#     x1 = BatchNormalization()(x1)
#     x1 = Dropout(0.2)(x1)
#     x1 = Activation('relu')(x1)
#     #second conv
#     x2 = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x1)
#     x2 = BatchNormalization()(x2)
#     x2 = Dropout(0.2)(x2)
#     #bypass with downsampling
#     xr = Conv2D(filters = filters, kernel_size = 1, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(residual)
#     xr = BatchNormalization()(xr)
#     xr = Dropout(0.2)(xr)
#     xr = AveragePooling2D((2,2))(xr)
#     #add together
#     x2 = Add()([x2, xr])
#     x2 = Activation('relu')(x2)
#     
#     #thrid conv
#     residual_2 = x2
#     x3 = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x2)
#     x3 = BatchNormalization()(x3)
#     x3 = Dropout(0.2)(x3)
#     x3 = Activation('relu')(x3)
#     #fourth conv
#     x4 = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x3)
#     x4 = BatchNormalization()(x4)
#     x4 = Dropout(0.2)(x4)
#     #identity bypass
#     x4 = Add()([x4, residual_2])
#     x4 = Activation('relu')(x4)
#     
#     #model_encoder_block = Model(inputs = inputs, outputs = x4)
#     return x4
# =============================================================================



def encoder_block(inputs, blocks_num, filters):
    return res_block(inputs, blocks_num, filters)



def decoder_block(inputs, filters):
    #dimensioanlity reduction
    x1 = Conv2D(filters = filters//4, kernel_size = 1, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Activation('relu')(x1)
    #upsampling
    x2 = Conv2DTranspose(filters = filters//4, kernel_size = 3, strides = 2, padding = 'same',kernel_initializer = 'he_normal')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Activation('relu')(x2)
    #go back to original dimensionality
    x3 = Conv2D(filters = filters, kernel_size = 1, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Activation('relu')(x3)
    #model_decoder_block = Model(inputs = inputs, outputs = x3)
    return x3


def dilation_block(inputs, filters, dropout_rate = 0.5):
    conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    
    merge1 = concatenate([inputs,conv1])
    conv2 = Dropout(dropout_rate)(merge1)
    conv2 = Conv2D(filters, 3, activation = 'relu', padding = 'same', dilation_rate = 2, kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    merge2 = concatenate([inputs, conv1, conv2])
    conv3 = Dropout(dropout_rate)(merge2)
    conv3 = Conv2D(filters, 3, activation = 'relu', padding = 'same', dilation_rate = 4, kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)


    merge3 = concatenate([inputs, conv1, conv2, conv3])
    conv4 = Dropout(dropout_rate)(merge3)
    conv4 = Conv2D(filters, 3, activation = 'relu', padding = 'same', dilation_rate = 8, kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    #merge6_4 = concatenate([pool5, conv6_1, conv6_2, conv6_3, conv6_4])
    #conv6_5 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate = 16, kernel_initializer = 'he_normal')(merge6_4)
    merge4 = concatenate([inputs, conv1, conv2, conv3, conv4])
    conv5 = Dropout(dropout_rate)(merge4)
    conv5 = Conv2D(filters, 3, activation = 'relu', padding = 'same', dilation_rate = 16, kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    
    merge5 = Add()([inputs, conv1, conv2, conv3, conv4, conv5])
    
    return merge5



def res_block(inputs, blocks_num, filters):
    merge = inputs
    x_skip = Conv2D(filters = filters, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    
    for i in range(blocks_num):
        if i == 0:
            x1 = Conv2D(filters = filters, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(merge)
            x1 = BatchNormalization()(x1)
            x1 = Dropout(0.2)(x1)
            x1 = Activation('relu')(x1)
        else:
            x1 = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(merge)
            x1 = BatchNormalization()(x1)
            x1 = Dropout(0.2)(x1)
            x1 = Activation('relu')(x1)
        x2 = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)
        x2 = Activation('relu')(x2)
        merge = Add()([x_skip, x2])
        x_skip = merge
    return merge


def dlinknet_simple_dilation(input_size = (512,512,3)):
    inputs = Input(shape = input_size)
    
    
    i1 = initial_block(inputs)
    
    e1 = encoder_block(i1, 3, 32)
    e2 = encoder_block(e1, 4, 64)
    e3 = encoder_block(e2, 6, 128)
    e4 = encoder_block(e3, 3, 256)
    
    #encoder_block1 = encoder_block(i1, 64)
    #e1 = encoder_block1.outputs
    
    #encoder_block2 = encoder_block(e1, 128)
    #e2 = encoder_block2.outputs
    
    #encoder_block3 = encoder_block(e2, 256)
    #e3 = encoder_block3.outputs
    
    #encoder_block4 = encoder_block(e3, 512)
    #e4 = encoder_block4.outputs
    
    e4 = dilation_block(e4, 256)
    
    
    #decoder
    d4 = decoder_block(e4, 128)
    d4 = Add()([e3, d4])
    
    d3 = decoder_block(d4, 64)
    d3 = Add()([e2, d3])
    
    d2 = decoder_block(d3, 32)
    d2 = Add()([e1, d2])
    
    d1 = decoder_block(d2, 16)
    d1 = Add()([i1, d1])
    
    #decoder_block4 = decoder_block(e4, 256)
    #d4 = decoder_block4.outputs
    #d4 = Add()([e3, d4])
    
    #decoder_block3 = decoder_block(d4, 128)
    #d3 = decoder_block3.outputs
    #d3 = Add()([e2, d3])
    
    #decoder_block2 = decoder_block(d3, 64)
    #d2 = decoder_block2.outputs
    #d2 = Add()([e1, d2])
    
    #decoder_block1 = decoder_block(d2, 64)
    #d1 = decoder_block1.outputs
    #d1 = Add()([i1, d1])
    
    #final upsampling
    f1 = Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = 'same',kernel_initializer = 'he_normal')(d1)
    f1 = BatchNormalization()(f1)
    f1 = Dropout(0.2)(f1)
    f1 = Activation('relu')(f1)
    
    #conv layer
    f2 = Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(f1)
    f2 = BatchNormalization()(f2)
    f2 = Dropout(0.2)(f2)
    f2 = Activation('relu')(f2)
    
    #logits
    logits = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(f2)
    outputs = Activation('sigmoid')(logits)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    model.summary()

    return model

dlinknet_simple_dilation()