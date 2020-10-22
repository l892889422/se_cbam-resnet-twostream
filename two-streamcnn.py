import math
import os
import keras
from keras import Model, models, Input
from keras.layers import Dense, Maximum
from keras_preprocessing.image import ImageDataGenerator
from numpy import concatenate


def cnn():
    img_input = keras.layers.Input(shape=(224,224,3))

    x = keras.layers.Conv2D(32, (3, 3), activation='relu',
                                 strides=(2, 2),
                                 padding='same',
                                 )(img_input)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)

    x = keras.layers.Conv2D(32,(3,3),
                      strides=(2, 2),
                            padding='same',
                      name='conv2')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)

    x = keras.layers.Conv2D(64, (3, 3),
                            strides=(2, 2),
                            padding='same',
                            )(x)
    x = keras.layers.BatchNormalization(name='bn_conv3')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)
    x = keras.layers.Flatten()(x)
###########################################下层网络###########################
    y = keras.layers.Conv2D(32, (5, 5), activation='relu',
                            padding='same',
                                 strides=(2, 2),
                                 )(img_input)
    y = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(y)

    y = keras.layers.Conv2D(32, (3, 3),
                            strides=(2, 2),
                            padding='same',
                            )(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(y)

    y = keras.layers.Conv2D(64, (3, 3),
                            strides=(2, 2),
                            padding='same',
                            )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')(y)
    y = keras.layers.Flatten()(y)
    #
    #
    # x = keras.layers.Reshape((2,-1))(x)
    # y = keras.layers.Reshape((2,-1))(y)
    ronghe = Maximum()([x, y])              #Add, Average

    #ronghe = concatenate((x, y))



    x = keras.layers.Dense(64)(ronghe)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    model = Model(inputs=img_input, outputs=x)

    return model

#######################################################main#################################################################
TRAIN_DIR='../data/train'
VALID_DIR='../data/validation'
# TRAIN_DIR='G:/python/untitled1/demo/data/train'
# VALID_DIR='G:/python/untitled1/demo/data/validation'
# SIZE = (224, 224)
BATCH_SIZE = 16       #每次送入的数据
# if __name__ == "__main__":
num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)]) #数量
num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

num_train_steps = math.floor(num_train_samples/BATCH_SIZE)    #样本/批数
num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)
model=cnn()

last = model.layers[-1].output  #输出
x = Dense(1, activation="sigmoid")(last)
model = Model(model.input, x)
model.summary()


model.compile(loss='binary_crossentropy',    #损失函数：对数损失
              optimizer='rmsprop',  #优化器
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,#归一化
    shear_range=0.2,#随机错位切换的角度
    zoom_range=0.2,#图片随机缩放的范围
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'G:/python/untitled1/demo/data/train',#图片地址
    target_size=(224,224),  # 调整图片大小为 150x150
    batch_size=16,  #设置批量数据的大小为32
    class_mode='binary')  #设置返回标签的类型 ，应对应损失函数
validation_generator = test_datagen.flow_from_directory(
    'G:/python/untitled1/demo/data/validation',
    target_size=(224,224),
    batch_size=16,
    class_mode='binary')
model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_steps,   #迭代进入下一轮次需要抽取的批次
        epochs=30,  #数据迭代的轮数
        validation_data=validation_generator,
        validation_steps=num_valid_steps)  #验证集用于评估的批次