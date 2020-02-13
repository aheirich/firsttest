#
# resnext time-to-accuracy-improvement tests
#

import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
import mycallbacks

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input

input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

model = ResNet50(
        input_tensor = input_tensor, include_top = False, weights = 'imagenet', 
        backend = tf.keras.backend, layers = tf.keras.layers, models = tf.keras.models, utils = tf.keras.utils)

learningRate = 0.01

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=learningRate), metrics=['accuracy'])

loss = model.evaluate(X, Y, verbose=1)
targetLoss = loss / 100.0

earlyStopping = mycallbacks.EarlyStopping(monitor='loss', baseline=targetLoss, verbose=1)

model.fit(X, Y, verbose=1, epochs=1000000, callbacks=[earlyStopping], validation_data=(X, Y))

