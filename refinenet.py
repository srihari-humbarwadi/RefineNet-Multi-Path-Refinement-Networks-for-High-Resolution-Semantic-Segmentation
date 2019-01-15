#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, Lambda, Add, Input, UpSampling2D
from tensorflow.keras.utils import plot_model


def BNConvUnit(tensor, filters, kernel_size=(3,3)):
    '''Returns Batch Normalized Conv Unit'''
    y = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', use_bias=False)(tensor)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y


def DeConvUnit(tensor, filters, stride=None):
    '''Returns Batch Normalized DeConv Unit'''
    y = Conv2DTranspose(filters=filters, kernel_size=(3,3), padding='same', strides=(stride,stride), kernel_initializer='he_normal', use_bias=False)(tensor)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y


def Upsample(tensor, method='transpose', scale=None):
    '''Upscaling feature map'''
    def bilinear_upsample(x, scale):
        dims = K.int_shape(x)
        resized = tf.image.resize_bilinear(images=x, size=[dims[1]*scale,dims[2]*scale])
        return resized
    
    if method == 'transpose':
        y = DeConvUnit(tensor=tensor, filters=128, stride=scale//2)
        y = BNConvUnit(tensor=y, filters=128)
        y = DeConvUnit(tensor=y, filters=64, stride=scale//2)
        y = BNConvUnit(tensor=y, filters=64)
        return y
    
    elif method == 'bilinear':
        y = Lambda(lambda x : bilinear_upsample(x, scale))(tensor)
        return y

def RCU(tensor, filters):
    '''Residual Conv Unit'''
    y = Activation('relu')(tensor)
    y = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(y)
    res = Add()([y, tensor])
    return res


def CRP(tensor, filters):
    '''Chained Residual Pooling Unit'''
    y = Activation('relu')(tensor)
    
    y1 = MaxPooling2D(pool_size=(5,5), padding='same', strides=1)(y)
    y1 = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(y1)
    y1_add = Add()([y, y1])
    
    y2 = MaxPooling2D(pool_size=(5,5), padding='same', strides=1)(y1)
    y2 = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer='he_normal')(y2)
    y2_add = Add()([y1_add, y2])
    
    return y2_add


def MRF(upper, lower, filters):
    '''Multi Resolution Fusion Unit'''
    if upper is None: # for  1/32 feature maps
        y = BNConvUnit(tensor=lower, filters=filters)
        return y
    y_lower = BNConvUnit(tensor=lower, filters=filters)
#     y_lower = Upsample(tensor=y_lower, method='bilinear', scale=2)
    y_lower = UpSampling2D(size=(2,2))(y_lower)
    y_upper = BNConvUnit(tensor=upper, filters=filters)
    y = Add()([y_lower, y_upper])
    return y


def RefineNetBlock(upper, lower):
    '''RefineNet Block for 1 resolution'''
    if lower is None: # for  1/32 feature maps
        y = RCU(tensor=upper, filters=512)
        y = RCU(tensor=y, filters=512)
        y = MRF(upper=None, lower=y, filters=512)
        y = CRP(tensor=y, filters=512)
        y = RCU(tensor=y, filters=512)
        return y
    y = RCU(tensor=upper, filters=256)
    y = RCU(tensor=y, filters=256)
    y = MRF(upper=y, lower=lower, filters=256)
    y = CRP(tensor=y, filters=256)
    y = RCU(tensor=y, filters=256)
    return y


def RefineNet(img_height, img_width, nclasses):
    '''Returns RefineNet model'''
    print('*** Building RefineNet Network ***')
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3), pooling=None)
    
    layer_names = ['activation_9', 'activation_21', 'activation_39', 'activation_48']
    filter_num = [512, 256, 256, 256]
    
    feature_maps = [base_model.get_layer(x).output for x in reversed(layer_names)]
    for i in range(4):
        feature_maps[i] = Conv2D(filter_num[i], 1, padding='same')(feature_maps[i])
    
    rf4 = RefineNetBlock(upper=feature_maps[0], lower=None)
    rf3 = RefineNetBlock(upper=feature_maps[1], lower=rf4)
    rf2 = RefineNetBlock(upper=feature_maps[2], lower=rf3)
    rf1 = RefineNetBlock(upper=feature_maps[3], lower=rf2)
    
    y = RCU(tensor=rf1, filters=256)
    y = RCU(tensor=y, filters=256)
    y = Upsample(tensor=y, scale=4)
    
    output = Conv2D(filters=nclasses, kernel_size=(1,1))(y)
    output = Activation('softmax', name='output_layer')(output)
    
    model = Model(inputs=base_model.input, outputs=output, name='RefineNet')
    print('*** Building Network Completed ***')
    print('*** Model Output Shape => ', model.output_shape, '***')
    return model
