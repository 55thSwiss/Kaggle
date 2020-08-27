'''
The data structure should be as follows:

root/
    originals/
        0/
            0_1.jpg
            0_2.jpg
            ...
        1/
            1_1.jpg
            1_2.jpg
            ...
        train/
            0/
                0_1.jpg
                0_2.jpg
                ...
            1/
                1_1.jpg
                1_2.jpg
                ...
        validation/
            0/
                0_1.jpg
                0_2.jpg
                ...
            1/
                1_1.jpg
                1_2.jpg
                ...
        test/
            0/
                0_1.jpg
                0_2.jpg
                ...
            1/
                1_1.jpg
                1_2.jpg
                ...
'''

import os

from keras import backend as K
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

cwd = os.getcwd()

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = (str(cwd) + r'\train\\')
validation_data_dir = (str(cwd) + r'\validation\\')
# find sample sizes
num_train_samples = 0
path, dirs, files = next(os.walk(train_data_dir))
for d in dirs:
    list_of_training_samples = os.listdir(train_data_dir + '\\' + d)
    num_train_samples = num_train_samples \
        + len(list_of_training_samples)
num_valid_samples = 0
path, dirs, files = next(os.walk(validation_data_dir))
for d in dirs:
    list_of_validation_samples = os.listdir(validation_data_dir + '\\' + d)
    num_valid_samples = num_valid_samples \
        + len(list_of_validation_samples)

# adjust these params
epochs = 40
batch_size = 50

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_valid_samples // batch_size)

model.save('chips.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training acc', color='blue')
plt.plot(epochs, val_acc, label='Validation acc', color='orange')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('Training and Validation Accuracy')

plt.figure()

plt.plot(epochs, loss, label='Training loss', color='blue')
plt.plot(epochs, val_loss, label='Validation loss', color='orange')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Training and Validation Loss')
