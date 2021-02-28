# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:49:51 2020

@author: Conor
"""


from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
import os
from keras.preprocessing.image import ImageDataGenerator
conv_base = VGG16(weights='imagenet',
    include_top=False,input_shape=(360, 208,3))

output = r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\run_dir'
train_dir = os.path.join(output,'train')
validation_dir = os.path.join(output,'validation')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(360, 208),
    batch_size=32,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(360, 208),
    batch_size=32,
    class_mode='categorical')

conv_base.trainable = False
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50)


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model.save(os.path.join(output,'videohiend.h5'))