from keras.models import Sequential
#from keras.layers import regularizers, MaxPooling2D, Convolution2D, TimeDistributed, LSTM
from keras.layers import Activation, Dropout, Flatten, GlobalAveragePooling2D, Dense
from keras import backend as K, metrics
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from keras.applications.xception import Xception
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
import pickle
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))
 
K.set_image_dim_ordering('tf')
nb_train_samples = 34400
nb_test_samples = 8600
tf.aggregation_method = 2
 
#X_train = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\X_Train.pkl', 'rb'))
#X_train = np.moveaxis(np.array(X_train), 0, -1)
 
#Y_train = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\Y_Train.pkl', 'rb'))
#Y_train = np.moveaxis(np.array(Y_train), 0, -1)
 
#X_test = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\X_Test.pkl', 'rb'))
#X_test = np.moveaxis(np.array(X_test), 0, -1)
 
#Y_test = pickle.load(open(r'C:\Users\Devin\Documents\GitHub\ASL_Data\Y_Test.pkl', 'rb'))
#Y_test = np.moveaxis(np.array(Y_test), 0, -1)
 
epochs = 50
batch_size = 16
nb_train_batches = nb_train_samples // batch_size
nb_test_batches = nb_test_samples // batch_size
nb_classes = 70
 
"""def preProcess (x, y, size):
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
print ("X_train bytes: ", X_train.nbytes)"""
 
base_model = Xception(include_top=False, weights='imagenet', input_shape=(200,200,3), pooling='max', classes=nb_classes)
 
x = base_model.output
x = Dropout(0.6)(x)
#x = Convolution2D(128, (3, 3),activation='relu')(x)
x = BatchNormalization()(x)
#x = Dropout(0.4)(x)
 
pred = Dense(nb_classes, activation='softmax')(x)
 
model = Model(inputs=base_model.input, outputs=pred)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy', metrics=['accuracy', top_k_categorical_accuracy])
 
train_datagen = ImageDataGenerator(
        rotation_range=0.2,
        shear_range=0.3,
        zoom_range=0.3,
        #horizontal_flip=True)
        )
test_datagen = ImageDataGenerator()
 
train_generator = train_datagen.flow_from_directory(
        r'C:\Users\Devin\Documents\GitHub\ASL_Data\Train_Data', 
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\Devin\Documents\GitHub\ASL_Data\Test_Data',
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='categorical')
 
model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_batches,
        epochs=12,
        validation_data=validation_generator,
        validation_steps=nb_test_batches)
model.save('ASL_Model.h5')
 
#model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_data=(X_test, Y_test), verbose=1)
#incorrects = np.nonzero(model.predict(X_test).reshape((-1,)) != Y_test)
#print ("The following examples were incorrectly classified")
#print (np.array(incorrects))
