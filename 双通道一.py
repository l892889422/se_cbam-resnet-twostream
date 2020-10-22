"""ResNet50 model for Keras.

@Auth：ljy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from keras.layers import Dense, Maximum
import os
import warnings

from . import get_submodules_from_kwargs
from . import imagenet_utils
from .imagenet_utils import decode_predictions
from .imagenet_utils import _obtain_input_shape
from keras import backend as K
preprocess_input = imagenet_utils.preprocess_input
import time
WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

backend = None
layers = None
models = None
keras_utils = None


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      padding='same',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##

    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      padding='same',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):

    def squeeze_excite_block(input, ratio=16):
        init = input
        # Compute channel axis
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        # Infer input number of filters
        filters = init._keras_shape[channel_axis]
        # Determine Dense matrix shape
        se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)

        se = layers.GlobalAveragePooling2D()(init)
        se = layers.Reshape(se_shape)(se)
        # se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        # se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal',
                           use_bias=False,name='rese'+str(time.time()))(se)
        se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
                           use_bias=False,name='rese'+str(time.time()))(se)
        x = layers.multiply([init, se])
        return x

    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##

    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##

    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      padding='same',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##

    x = layers.Activation('relu')(x)
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             padding='same',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = layers.MaxPooling2D(strides=8, padding='same')(shortcut)##

    x = squeeze_excite_block(x)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    first = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1')(first)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same',)(x)

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
################################################3*3卷积核#######################################################

    y = layers.Conv2D(64, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1')(first)
    y = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(y)
    y = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)##

    y = layers.Activation('relu')(y)
    y = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(y)
    y = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(y)

    y = conv_block(y, 3, [64, 64, 256], stage=2, block='a1', strides=(1, 1))
    y = identity_block(y, 3, [64, 64, 256], stage=2, block='b1')
    y = identity_block(y, 3, [64, 64, 256], stage=2, block='c1')

    y = conv_block(y, 3, [128, 128, 512], stage=3, block='a1')
    y = identity_block(y, 3, [128, 128, 512], stage=3, block='b1')
    y = identity_block(y, 3, [128, 128, 512], stage=3, block='c1')
    y = identity_block(y, 3, [128, 128, 512], stage=3, block='d1')

    y = conv_block(x, 3, [256, 256, 1024], stage=4, block='a1')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='b1')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='c1')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='d1')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='e1')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='f1')

    y = conv_block(y, 3, [512, 512, 2048], stage=5, block='a1')
    y = identity_block(y, 3, [512, 512, 2048], stage=5, block='b1')
    y = identity_block(y, 3, [512, 512, 2048], stage=5, block='c1')


    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)##
    y = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)##
    # y = layers.Flatten()(y)
    # x = layers.Flatten()(x)
    x = layers.Maximum()([x, y])
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = models.Model(inputs, x, name='resnet50')

    # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = keras_utils.get_file(
    #             'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
    #             WEIGHTS_PATH,
    #             cache_subdir='models',
    #             md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    #     else:
    #         weights_path = keras_utils.get_file(
    #             'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #             'D:/AI1403/ljy/预训练',
    #             cache_subdir='models',
    #             md5_hash='a268eb855778b3df3c7506639542a6af')
    #     model.load_weights(weights)
    #     if backend.backend() == 'theano':
    #         keras_utils.convert_all_kernels_in_model(model)
    # elif weights is not None:
    #     model.load_weights(weights)
    model.load_weights('D:/AI1403/ljy/预训练/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
    return model
