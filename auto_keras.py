# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:50:18 2019

@author: khira
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input as preprocess_resnet50
# import the necessary packages
from sklearn.metrics import classification_report
from keras.datasets import cifar10
import autokeras as ak
import os
import numpy as np

train_path = 'data/train/'
valid_path = 'data/valid/'
test_path = 'data/test/'

#trainGen = ImageDataGenerator(rescale=1./255,
#                            shear_range=0.2,
#                            zoom_range=0.2,
#                            horizontal_flip=True,
#                            preprocessing_function=preprocess_resnet50)
#
#trainData = trainGen.flow_from_directory(train_path,
#                                   target_size=(224,224),
#                                   color_mode='rgb',
#                                   batch_size=32,
#                                   class_mode='categorical',
#                                   shuffle=True)

# initialize the output directory
OUTPUT_PATH = "output/"
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


# initialize the list of training times that we'll allow
# Auto-Keras to train for
TRAINING_TIMES = np.array([4, 8, 16, 24])*3600

# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]

modelSearches = {}
# loop over the number of seconds to allow the current Auto-Keras
# model to train for
for seconds in TRAINING_TIMES:
    # train our Auto-Keras model
    print("[INFO] training model for {} seconds max...".format(
        seconds))
    model = ak.ImageClassifier(verbose=False)
    try:
        model.fit(trainX, trainY, time_limit=seconds)
    except TimeoutError:
        modelSearches[seconds] = ['failed', model]
        print("TimeoutError: couldn't find a model in the given time:", seconds/3600,"hrs")
        continue
    print("Found a model in the given time:", seconds/3600,"hrs")
    modelSearches[seconds] = ['success', model]
    model.final_fit(trainX, trainY, testX, testY, retrain=True)

    # evaluate the Auto-Keras model
    score = model.evaluate(testX, testY)
    predictions = model.predict(testX)
    report = classification_report(testY, predictions,
        target_names=labelNames)

    # write the report to disk
    p = os.path.sep.join(OUTPUT_PATH, "{}.txt".format(seconds))
    f = open(p, "w")
    f.write(report)
    f.write("\nscore: {}".format(score))
    f.close()