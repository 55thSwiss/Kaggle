
import os

from keras import backend as K
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing import image

import numpy as np

cwd = os.getcwd()

img_width, img_height = 150, 150

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

model = load_model('chips.h5')

test_dir = (r'__your test directory here__')

for filename in os.listdir(test_dir):
    imagePath = str(test_dir) + filename

    test_image = image.load_img(imagePath, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # predict the result
    result = model.predict(test_image)
    print(result)
