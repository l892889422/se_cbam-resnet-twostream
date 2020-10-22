from keras import models
from keras.layers import concatenate, Flatten, GlobalAveragePooling2D

import up_model,down_model

def two_stream_model(include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=5,
                        ):
    model_up=up_model(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg', classes=5)
    model_down=down_model(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg', classes=5)
    x=model_up.layers[-1].output
    y=model_down.layers[-1].output
    x=Flatten(x)
    y = Flatten(y)
    feature = concatenate([x, y])
    shuchu=GlobalAveragePooling2D()(feature)
    model = models.Model(feature, shuchu, name='twostream_resnet50_se')
    return model