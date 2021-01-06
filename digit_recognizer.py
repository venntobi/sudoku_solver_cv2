#%%
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import random 
import cv2 
from scipy import ndimage
from tensorflow.python.keras.layers.core import Dropout

num_classes = 9
batch_size = 256
epochs = 20

img_shape = 28

DIR = "Images"
CATEGORIES = [str(x) for x in range(1,10)]


def get_best_shift(img):
  cy, cx = ndimage.measurements.center_of_mass(img)
  rows, cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx, shifty

def shift(img, sx, sy):
  rows, cols = img.shape
  M = np.float32([[1,0,sx],[0,1,sy]])
  shifted = cv2.warpAffine(img,M,(cols,rows))
  return shifted

def center_img(img):
  img = cv2.bitwise_not(img)
  shiftx, shifty = get_best_shift(img)
  shifted = shift(img, shiftx, shifty)
  img = shifted
  img = cv2.bitwise_not(img)

  return img 

training_data = []

def create_training_data():
  for cat in CATEGORIES:
    path = os.path.join(DIR, cat)
    class_num = CATEGORIES.index(cat)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
      new_array = cv2.resize(img_array, (img_shape, img_shape))
      new_array = center_img(new_array)
      training_data.append([new_array, class_num])

create_training_data()

random.seed(123)
random.shuffle(training_data)


x_train = []
y_train = []
x_test = []
y_test = []

for i in range(len(training_data)*8//10):
  x_train.append(training_data[i][0])
  y_train.append(training_data[i][1])

for i in range(len(training_data)*8//10, len(training_data)):
  x_test.append(training_data[i][0])
  y_test.append(training_data[i][1])


x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], img_shape, img_shape,1)
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], img_shape, img_shape,1)
input_shape = (img_shape,img_shape,1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

y_train = np.array(y_train)
y_test = np.array(y_test)


def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3),
                        activation="relu",
                        input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model


model = create_model()

model.fit(x_train, y_train,
          epochs=epochs,
          verbose=1,
          batch_size=batch_size,
          validation_data=(x_test, y_test))

model.save("mnist_model")