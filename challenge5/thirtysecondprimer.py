# from Keras Documentation - 30 second primer
# https://keras.io

import keras
from keras.models import Sequential

model = Sequential()

from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))



from dir_helpers import *
from image_helpers import *
import numpy as npy

NUM_IMAGES = 20

def dummyProcess(image):
    # return image.histogram()
    d = image.getdata()
    #print("Image data is", len(d), "x", len(d[0]))
    new_d = npy.array(d)
    # new_d = npy.reshape(new_d, [16384*3])
    # print("Image data is", len(new_d), "x 1")
    return new_d

def loadImagesForSciKit(input_dir, process, max_per_dir=9999):
    all_images = loadImagesRecursive(input_dir, max_per_dir=200)
    output_data = []
    classes = []

    for image in all_images:
        # print("from:", image.parent.name + '/' + image.name)
        img = loadImage128x128(image)
        # print("loaded:", img)
        image_data = process(img)
        #filename = output_dir + '/' + str(image_number) + '.jpg'
        dirname = image.parent.name  # class
        # print("Image class ", dirname)
        output_data.append(image_data)
        classes.append(dirname)
    return output_data, classes

print("Loading images")
# d, c = loadImagesForSciKit('../processed_images', dummyProcess, NUM_IMAGES)
d, c = loadImagesForKeras('../processed_images', dummyProcess, NUM_IMAGES)
print("Done")

num_images = len(d)
num_pixels = len(d[0])
num_layers = len(d[0][0])
print("got", num_images, "x", num_pixels, "x", num_layers)
values = npy.reshape(d, [num_images*num_pixels*num_layers])
print("data range", max(values), "to", min(values))



'''
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
model.train_on_batch(x_batch, y_batch)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

classes = model.predict(x_test, batch_size=128)
'''