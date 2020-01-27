#
# firsttest
#

from sys import argv
from tensorflow import keras
from timeit import default_timer as timer
import tensorflow as tf

mnistSize = 28
tf.keras.backend.set_floatx('float32')
batchSize = 128
print("argv:", argv)

#
# testDepth
#

def sampleOneConfiguration(depth, width, x_test, y_test):
    model = keras.models.Sequential()
    numNeurons = 0
    numWeights = 0
    model.add(keras.layers.Flatten(input_shape = (mnistSize, mnistSize)))
    for i in range(depth):
        model.add(keras.layers.Dense(width))
        numNeurons = numNeurons + width
        numWeights = numWeights + width * (mnistSize * mnistSize)
    model.add(keras.layers.Dense(width))
    numNeurons = numNeurons + width
    numWeights = numWeights + width * width
    model.add(keras.layers.Dense(1))
    numNeurons = numNeurons + 1
    numWeights = numWeights + width
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
    start = timer()
    score = model.evaluate(x_test, y_test, verbose = 2, batch_size = batchSize)
    end = timer()
    elapsed = end - start
    return (elapsed, numNeurons, numWeights)

#
# use mnist data set, doesn't really matter
#

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#
# warm up
#

depth = int(argv[1])
width = int(argv[2])

print("warmup")
numWarmups = 3
for i in range(numWarmups):
    (t, n, w) = sampleOneConfiguration(depth, width, x_test, y_test)
    if t is None:
        sys.exit()
print("")

print("sample")
(time, numNeurons, numWeights) = sampleOneConfiguration(depth, width, x_test, y_test)
print("Elapsed seconds =", time)
print("numNeurons =", numNeurons)
print("numWeights =", numWeights)
