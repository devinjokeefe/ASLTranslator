from keras.models import Sequential
from keras.layers import regularizers, MaxPooling2D, Convolution2D, TimeDistributed, LSTM
from keras.layers import Activation, Dropout, Flatten, GlobalAveragePooling2D, Dense
from keras import applications, backend as K, metrics
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.applications.resnet50 import ResNet50
import pickle
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

K.set_image_dim_ordering('tf')
nb_train_samples = 32900
nb_test_samples = 8600
tf.aggregation_method = 2

X_train = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\X_Train.pkl', 'rb'))
#X_train = np.moveaxis(np.array(X_train), 0, -1)

Y_train = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\Y_Train.pkl', 'rb'))
#Y_train = np.moveaxis(np.array(Y_train), 0, -1)

X_test = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\X_Test.pkl', 'rb'))
#X_test = np.moveaxis(np.array(X_test), 0, -1)

Y_test = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\Y_Test.pkl', 'rb'))
#Y_test = np.moveaxis(np.array(Y_test), 0, -1)

epochs = 50
batch_size = 1
nb_train_batches = nb_train_samples // batch_size
nb_test_batches = nb_test_samples // batch_size
nb_classes = 70

def preProcess (x, y, size):
    channel = np.ones(1)
    x = np.array(x).reshape(size//1, 200, 200, 3)
    y = to_categorical(y, num_classes=nb_classes)

    return x, y

X_train, Y_train = preProcess(X_train, Y_train, nb_train_samples)
X_test, Y_test = preProcess(X_test, Y_test, nb_test_samples)

print('X_train shape: ', np.array(X_train).shape)
print('Y_train shape: ', np.array(Y_train).shape)
print('X_test shape: ', np.array(X_test).shape)
print('Y_test shape: ', np.array(Y_test).shape)
print ("X_train bytes: ", X_train.nbytes)

def CNN_LSTM ():
    model = Sequential()

    model.add(TimeDistributed(Convolution2D(32, (3, 3), activation='relu', padding='same'), input_shape=(10, 200, 200, 1)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Convolution2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Dropout(0.35)))
    
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(128, return_sequences=True))
    model.add(Activation('relu'))
    
    model.add(LSTM(256, return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dropout(0.35))
    
    model.add(Dense(nb_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy', metrics.top_k_categorical_accuracy])

    model.fit(X_train, Y_train, batch_size=1, epochs=225, validation_data=(X_test, Y_test), verbose=1)

def CNN ():
    print ("Beginning Convolutional Neural Network")
    model = Sequential()
    
    model.add(Convolution2D(4, (3, 3), activation='relu', padding='same', input_shape=(200,200,3)))
    model.add(Convolution2D(4, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))
#    model.add(Convolution2D(2, (3, 3), activation='relu', padding='same', input_shape=(200,200,1)))
 #   model.add(Convolution2D(4, (3, 3), activation='relu', padding='same'))
  #  model.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
   # model.add(MaxPooling2D(2, 2))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same', input_shape=(200,200,3)))
    model.add(Convolution2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Convolution2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', metrics.top_k_categorical_accuracy])
    model.summary()
    model.fit(X_train, Y_train, batch_size=256, epochs=225, validation_data=(X_test, Y_test), verbose=1)    
CNN()
def trainOnBatch ():
    for e in range(0, epochs):
        topAcc = 0.0
        topFiveAcc = 0.0
        i = 0
        for i in range(0, nb_train_batches):
    
            start = i * batch_size
            end = start + batch_size
    
            channel = np.ones(1)
            x_train = np.array(X_train[start:end][:][:])
            x_train = x_train.reshape(batch_size, 300, 300, 1)
    
            y_train = np.array(Y_train[start:end][:])
            y_train = to_categorical(y_train, num_classes=nb_classes)
    
            model.train_on_batch(x_train, y_train)
            i += 1
            if (i % 21 == 0):
                print(i//21, "% done")
     
        for i in range(0, nb_test_batches):
    
            start = i * batch_size
            end = start + batch_size
    
            channel = np.ones(1)
    
            x_test = np.array(X_test[start:end][:][:])
            x_test = x_test.reshape(batch_size, 300, 300, 1)
    
            y_test = np.array(Y_test[start:end][:])
            y_test = to_categorical(y_test, num_classes=nb_classes)
    
            score = model.evaluate(x_test, y_test, batch_size=batch_size)
            topAcc += score[2]
            topFiveAcc += score[3]
         
    
        print("Top 1 accuracy: ", topAcc // nb_test_batches)
        print("Top 5 accuracy: ", topFiveAcc // nb_test_batches)
         
        model.summary()
