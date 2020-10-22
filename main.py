"""
This is a revised implementation from Cifar10 ResNet example in Keras:
(https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
"""

from __future__ import print_function

import math

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras import  optimizers
#import keras_applications.ResNet50V2
#import keras.applications.resnet_v2
from keras.datasets import cifar10
from models import resnext, resnet_v1, resnet_v2, mobilenets, inception_v3, inception_resnet_v2, densenet
from utils import lr_schedule
import numpy as np
import os
# loss: 0.9184 - accuracy: 0.8234 - val_loss: 6522.6538 - val_accuracy: 0.0778 batch_size = 64
# Training parameters
SIZE = (224, 224)
batch_size = 64
epochs = 200
data_augmentation = False   #  数据扩充
num_classes = 5       # 十分类
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy减像素平均值提高accu
base_model = 'resnet50'
# Choose what attention_module to use: cbam_block / se_block / None
attention_module = 'cbam_block'
# attention_module = None
model_type = base_model if attention_module==None else base_model+'_'+attention_module


TRAIN_DIR='D:/AI1403\ljy/CBAM-keras-master/数据集/train'
VALID_DIR='D:/AI1403\ljy/CBAM-keras-master/数据集/val'

num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)]) #数量
num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

num_train_steps = math.floor(num_train_samples/batch_size)    #样本/批数
num_valid_steps = math.floor(num_valid_samples/batch_size)
val_gen = keras.preprocessing.image.ImageDataGenerator()
#val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)    #水平垂直旋转
gen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
#以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
#路径、图像尺寸、返回标签数组的形式、是否打乱数据、batch数据的大小，默认为32
batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=batch_size)
val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=batch_size)
input_shape =(224,224,3)




depth = 50 # For ResNet, specify the depth (e.g. ResNet50: depth=50)
model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth, attention_module=attention_module)
#model = resnet_v2.resnet_v2(input_shape=input_shape, depth=depth, attention_module=attention_module)
#model = resnext.ResNext(input_shape=input_shape, classes=num_classes, attention_module=attention_module)

classes = list(iter(batches.class_indices))  # 用训练好的模型预测时，预测概率序列和Labels的对应关系
model.layers.pop()  # 弹出模型的最后一层

# 一个层意味着将其排除在训练之外，即其权重将永远不会更新。
for layer in model.layers:
    layer.trainable = False

last = model.layers[-1].output  # 输出
# 全连接层 神经元数量和激活函数
print('神经元数量', len(classes))
x = Dense(len(classes), activation="softmax")(last)
finetuned_model = Model(model.input, x)

# 设置损失函数，优化器，模型在训练和测试时的性能指标
finetuned_model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])#Adam(lr=lr_schedule(0)
for c in batches.class_indices:
    classes[batches.class_indices[c]] = c
finetuned_model.classes = classes
# 早停法防止过拟合，patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
#early_stopping = EarlyStopping(patience=10)
checkpointer = ModelCheckpoint('resnet50_cbam.h5', verbose=1, save_best_only=True)  # 添加模型保存点
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

callbacks = [checkpointer, lr_reducer, lr_scheduler]
# 拟合模型
finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=60,
                              callbacks=callbacks, validation_data=val_batches,
                              validation_steps=num_valid_steps)
finetuned_model.save('resnet50_cbam_final.h5')

