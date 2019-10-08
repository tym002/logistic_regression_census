#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:10:33 2019

@author: tianyu
"""

from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras import optimizers,regularizers
import numpy as np
import matplotlib.pyplot as plt


X_train = np.transpose(np.load('train_data.npy'))
Y_train = np.load('train_label_keras.npy')
X_test = np.transpose(np.load('test_data.npy'))
Y_test = np.load('test_label_keras.npy')

model = Sequential() 
model.add(Dense(1,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('sigmoid')) 
batch_size = 100
nb_epoch = 10
model.compile(optimizer=optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['binary_accuracy']) 
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test)) 
score = model.evaluate(X_test, Y_test, verbose=0) 

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

print('Test score:', score[0]) 
print('Test accuracy:', score[1])