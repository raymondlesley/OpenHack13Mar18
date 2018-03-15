# CNN model - parameterised model

import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K

def RunCNNmodel(x_train, y_train, x_test, y_test, input_shape, num_images, num_classes, num_epochs, feature_dim, initial_conv_size, second_cov_size, first_dropout, second_dropout, dense_size):
    feature_size = (feature_dim, feature_dim)
    batch_size = num_images

    model = Sequential()

    # original layers

    model.add(Conv2D(initial_conv_size, feature_size,  # was 32
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(second_cov_size, feature_size, activation='relu'))  # was 64
    model.add(MaxPooling2D(pool_size=(2, 2)))  # was (2, 2)
    model.add(Dropout(first_dropout))  # was 0.25
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))   # was Dense(128, ...)
    model.add(Dropout(second_dropout))  # was 0.5
    model.add(Dense(num_classes, activation='softmax'))

    '''
    # experiment
    model.add(Conv2D(32, feature_size,
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, feature_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # was (2, 2)
    
    #model.add(Dropout(0.25))  # was 0.25
    #model.add(Flatten())
    model.add(Conv2D(64, feature_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # was (2, 2)
    
    #model.add(Dense(64, activation='relu'))   # was Dense(128, ...)
    #model.add(Dropout(0.25))  # was 0.5
    model.add(Dense(num_classes, activation='softmax'))
    '''

    #model.compile(loss=keras.losses.categorical_crossentropy,
    #              optimizer=keras.optimizers.Adadelta(),  # try "rmsprop"
    #              metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="rmsprop",
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=0,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    loss = score[0]
    accuracy = score[1] * 100.0
    #print('Test loss: %2.1f' % loss)
    #print('Test accuracy:  %2.1f%%' % accuracy)
    #print(score[1]*100)
    return accuracy
