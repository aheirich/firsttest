#
# resnet time-to-accuracy-improvement tests
#

import os
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import numpy
import time
print("Tensorflow version " + tf.__version__)


TPU = True
if TPU:
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver("tpuvm1")  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
  except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
else:
  print("gpu tbd")

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import mycallbacks

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

preprocessingTime = 0

def timedPreprocessInput(x, **kwargs):
  t0 = time.perf_counter_ns();
  preprocess_input(x, **kwargs)
  t1 = time.perf_counter_ns();
  elapsed_ns = t1 - t0
  global preprocessingTime
  preprocessingTime = preprocessingTime + elapsed

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


def fitModel(weights, batchSize, improvement, learningRate):
  model = makeModel(weights, learningRate)
  train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

  image_folder = "/home/alan_heirich/firsttest/imagenet"
  train_generator=train_datagen.flow_from_directory(image_folder,
                                                   target_size=(224,224),
                                                   color_mode='rgb',
                                                   batch_size=batchSize,
                                                   class_mode='categorical',
                                                   shuffle=True)
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


# display device type
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from sys import argv
batchSize = int(argv[1])

#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
#tf.config.experimental_connect_to_host(resolver.master())
#tf.tpu.experimental.initialize_tpu_system(resolver)
#strategy = tf.distribute.experimental.TPUStrategy(resolver)

#with strategy.scope():
if True:
# warmup
  fitModel(None, batchSize, 1000.0, 0.01)


  preprocessingTime = 0
  t0 = time.perf_counter_ns()
  fitModel(None, batchSize, 1000.0, 0.01)
  fitModel('imagenet', batchSize, 10.0, 0.0001)
  t1 = time.perf_counter_ns()
  elapsed_ns = t1 - t0
  processingTime = elapsed_ns - preprocessingTime
  print(">> RESNET50 batch size", batchSize, "elapsed ns", elapsed_ns,
     "preprocessing", preprocessingTime, "processing", processingTime)
#  (elapsed_ns, preprocessingTime, processingTime)


