
import sys
from subprocess import Popen, PIPE
import numpy as np
from scipy.stats import linregress

numSamples =  5


def runTest(argv):
    print(argv)
    process = Popen(argv, stdout=PIPE, stderr=PIPE, shell=False)
    stdout, stderr = process.communicate()
    out = stdout.decode('UTF-8')
    error = stderr.decode('UTF-8')
    result = out + error
    divider = "---------------------------------------"
    print(divider)
    print(result)
    print(divider)
    return result

def filterOutput(text):
    lines = text.split("\n")
    result = [ None, None, None ]
    index = 0
    for i in range(len(lines)):
        for key in ["Elapsed seconds =", "numNeurons =", "numWeights ="]:
            if(lines[i].startswith(key)):
                datum = float(lines[i][len(key):])
                result[index] = datum
                index = index + 1
    return result


#
# test time to accuracy improvement for increasing bbatch sizes
#

print("testing time to accuracy improvement for increasing batch sizes")
batchCurves = []
processingCurves = []
preprocessingCurves = []

for i in range(numSamples):
    print("batch pass", i)
    batchCurve = []
    processingCurve = []
    preprocessingCurve = []
    batchSize = 1
    time = 0
    while time is not None:
        print("test batch size", batchSize)
        stdout = runTest(["python3", "resnet_time_accuracy_improvement.py", str(batchSize)])
        (time, preprocessingTime, processingTime) = filterOutput(stdout)
        if time is not None:
            batchCurve.append(time)
            processingCurve.append(processingTime)
            preprocessingCurve.append(preprocessingTime)
            batchSize = batchSize * 2
    batchCurves.append(batchCurve)
    processingCurves.append(processingCurve)
    preprocessingCurves.append(preprocessingCurve)

print("batch curves", batchCurves)
print("preprocessing", preprocessingCurves)
print("processing", processingCurves)


#  load_infer

print("testing depth")
depthCurves = []
for i in range(numSamples):
    print("depth pass", i)
    depthCurve = []
    numHiddenLayers = 1
    time = 0
    depthNeuronAxis = []
    depthWeightAxis  = []
    while time is not None:
        print("test", numHiddenLayers, "X 8")
        stdout = runTest(["python3", "/load_infer.py", str(numHiddenLayers), "8"])
        (time, numNeurons, numWeights) = filterOutput(stdout)
        if time is not None:
            depthCurve.append(time)
            depthNeuronAxis.append(numNeurons)
            depthWeightAxis.append(numWeights)
            numHiddenLayers = numHiddenLayers * 2
    depthCurves.append(depthCurve)

#
# test increasing widths
#

print("testing width")
widthCurves = []
for i in range(numSamples):
    print("width pass", i)
    widthCurve = []
    width = 8
    time = 0
    widthNeuronAxis = []
    widthWeightAxis  = []
    while time is not None:
        print("test 1 X", width)
        stdout = runTest(["python3", "load_infer.py", "1", str(width)])
        (time, numNeurons, numWeights) = filterOutput(stdout)
        if time is not None:
            widthCurve.append(time)
            widthNeuronAxis.append(numNeurons)
            widthWeightAxis.append(numWeights)
            width = width * 2
    widthCurves.append(widthCurve)



#
# report
#

print("depth curves", depthCurves)
print("")

print("width curves", widthCurves)
print("")


depthMean = np.sum(depthCurves, 0) / numSamples
widthMean = np.sum(widthCurves, 0) / numSamples

print("depthMean =", depthMean)
print("widthMean =", widthMean)
print("depthNeuronAxis =", depthNeuronAxis)
print("depthWeightAxis =", depthWeightAxis)
print("widthNeuronAxis =", widthNeuronAxis)
print("widthWeightAxis =", widthWeightAxis)

print("np.log(depthMean) =", np.log(depthMean))
logDepthVsNeurons = linregress(np.log(depthMean),  np.log(depthNeuronAxis))
print("logDepthVsNeurons =", logDepthVsNeurons)

depthVsNeurons = linregress(depthMean, depthNeuronAxis)
depthVsWeights = linregress(depthMean, depthWeightAxis)

widthVsNeurons = linregress(widthMean, widthNeuronAxis)
widthVsWeights = linregress(widthMean, widthWeightAxis)

print("depthVsNeurons =", depthVsNeurons)
print("depthVsWeights =", depthVsWeights)
print("widthVsNeurons =", widthVsNeurons)
print("widthVsWeights =", widthVsWeights)
