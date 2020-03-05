#
# resnet time-to-accuracy-improvement tests
#

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import mycallbacks

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model


def makeModel(weights, learningRate):

  input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
  model = ResNet50(
          input_tensor = input_tensor, include_top = True, weights = weights, 
          backend = tf.keras.backend, layers = tf.keras.layers, 
          models = tf.keras.models, utils = tf.keras.utils)
  model.compile(loss=tf.keras.losses.categorical_crossentropy, 
               optimizer=tf.keras.optimizers.Adam(lr=learningRate), 
               metrics=['accuracy'])
  plot_model(model, to_file='model.png')
  return model


def fitModel(weights, train_generator, improvement, learningRate):
  model = makeModel(weights, learningRate)
  loss = model.evaluate(train_generator, verbose=1)[0]
  print("initial loss", loss)
  targetLoss = loss / improvement
  print("target loss", targetLoss)
  earlyOut = mycallbacks.EarlyOut(monitor='loss', baseline=targetLoss, verbose=1)
  step_size_train=train_generator.n//train_generator.batch_size
  model.fit_generator(generator=train_generator,
                     steps_per_epoch=step_size_train,
                     callbacks=[earlyOut],
                     epochs=100000)



train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

image_folder = '/home/aheirich/Documents/SLAC/software/MLPerf/firsttest/imagenet/'
train_generator=train_datagen.flow_from_directory(image_folder,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

fitModel(None, train_generator, 1000.0, 0.01)
fitModel('imagenet', train_generator, 10.0, 0.0001)

