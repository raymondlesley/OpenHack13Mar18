# from Keras Documentation - 30 second primer
# https://keras.io

import keras
from keras.models import Sequential

model = Sequential()

'''
# example code
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
'''


# image import

from dir_helpers import *
from image_helpers import *
import numpy as npy

NUM_IMAGES = 20

def dummyProcess(image):
    # return image.histogram()
    d = image.getdata()
    #print("Image data is", len(d), "x", len(d[0]))
    new_d = npy.array(d)
    new_d = npy.reshape(new_d, [128, 128, 3])
    # print("Image data is", len(new_d), "x 1")
    return new_d

def loadImagesForKeras(input_dir, process, max_per_dir=9999):
    all_images = loadImagesRecursive(input_dir, max_per_dir)
    output_data = []
    classes = []

    for image in all_images:
        print("from:", image.parent.name + '/' + image.name)
        img = loadImage128x128(image)
        # print("loaded:", img)
        image_data = process(img)
        print("data is", image_data.shape)
        #filename = output_dir + '/' + str(image_number) + '.jpg'
        dirname = image.parent.name  # class
        print("Image class ", dirname)
        output_data.append(image_data)
        classes.append(dirname)

    return output_data, classes

print("Loading images")
d, c = loadImagesForKeras('../processed_images', dummyProcess, NUM_IMAGES)
print("Done")

from classes import *
print("tents =", getClassIdx("tents"))

class_list = []
for cls in c:
    idx = getClassIdx(cls)
    class_list.append(idx)

num_images = len(d)
img_width = len(d[0])
img_height = len(d[0][0])
num_layers = len(d[0][0][0])
print("got", num_images, "x", img_width, "x", img_height, "x", num_layers)
values = npy.reshape(d, [num_images*img_width*img_height*num_layers])
print("data range", max(values), "to", min(values))
print("%d classes" % len(class_list))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(npy.asarray(d), npy.asarray(class_list), test_size=0.33, random_state=42)

'''
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train, y_train, epochs=5, batch_size=32)
# model.train_on_batch(x_batch, y_batch)

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)

classes = model.predict(X_test, batch_size=128)
'''

### #####################################
#
# Keras CNN section - inspired by https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = num_images # 128
num_classes = 12 # 10
epochs = 12

# input image dimensions
img_rows, img_cols = 128, 128
input_shape = X_test[0].shape # (img_rows, img_cols, 1)

x_train = X_train.astype('float32')
x_test = X_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(y_train.shape[0], 'train classes')
print(y_test.shape[0], 'test classes')
print("num_classes =", num_classes)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
