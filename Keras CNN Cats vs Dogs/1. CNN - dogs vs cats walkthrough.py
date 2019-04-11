'''
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
'''

import os
import shutil

'''
1. COPYING IMAGES TO TRAINING, VALIDATION, AND TEST DIRECTORIES
'''
'''
setup the directories
'''
# working directory
cwd = os.getcwd()

# image database folder
folder = (r'\dogs-vs-cats\\')

# original dataset directory
oringal_database_dir = (str(cwd) + folder + r'\train\\')

# directory to store the smaller dataset (sample)
base_dir = (str(cwd) + r'\dogs-vs-cats_small\\')
try:
    os.mkdir(base_dir)
except:
    pass

    # directories for the training, validation, and test splits
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
try:
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
except:
    pass

# train data folders
train_cats_dir = os.path.join(train_dir, 'cats')

train_dogs_dir = os.path.join(train_dir, 'dogs')
try:
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)
except:
    pass

# validation data folders
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
try:
    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)
except:
    pass

# test data folders
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
try:
    os.mkdir(test_cats_dir)
    os.mkdir(test_dogs_dir)
except:
    pass

'''
segregate the photos
'''
try:
    # copy the first 1000 cat miages to train_cats_dir
    fnames = [f'cat.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(oringal_database_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    # copies the next 500 to validation_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(oringal_database_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    # copies the next 500 to test_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(oringal_database_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    # copy the first 1000 dog miages to train_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(oringal_database_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # copies the next 500 to validation_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(oringal_database_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # copies the next 500 to test_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(oringal_database_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
except:
    pass

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test_dogs images:', len(os.listdir(test_dogs_dir)))

'''
2. INSTANTIATING A CONVNET (CREATING A MODEL)
'''

from keras import layers
from keras import models


# this model was used on a 4000 original image binary
# classification problem. 1000 train, 500 validation,
# 500 test for each example.

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(129, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(129, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# a look at how the dimensions of the feature map change in each layer
model.summary()

from keras import optimizers

# configure the model for training
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

'''
3. READ IMAGES FROM THE DIRECTORY
'''

from keras.preprocessing.image import ImageDataGenerator

# rescale the images by 255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# use Image Data Generator to read images from directories
train_generator = train_datagen.flow_from_directory(
    # target directory
    train_dir,
    # resize all images to 150 x 150
    target_size=(150, 150),
    batch_size=20,
    # because of using binary crossentropy loss, you need binary labels
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape: ', data_batch.shape)
    print('labels batch shape: ', labels_batch.shape)
    break

'''
5.8 Fitting the model using a batch generator
'''

history = model.fit_generator(
    train_generator,
    # batch size was 20 samples, so it will take 100
    # batches before the model sees the 2000 photos
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

'''
5.9 Save the model
'''

model.save('cats_and_dogs_small_1.h5')