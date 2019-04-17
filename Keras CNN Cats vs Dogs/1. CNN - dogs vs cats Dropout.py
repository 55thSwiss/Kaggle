'''
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
'''

import os
import shutil

'''
5.4 Copying images to training, validation, and test directories
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
5.5 Instantiating a small convnet for dogs vs. cats classification
'''

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
# new line for Dropout
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# a look at how the dimensions of the feature map change in each layer
model.summary()

'''
5.6 Configure the model for training
'''

from keras import optimizers

# configure the model for training
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

'''
5.7 Using ImageDataGenerator to read images from directories
'''

from keras.preprocessing.image import ImageDataGenerator

# rescale the images by 255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# # use Image Data Generator to read images from directories
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

'''
5.10 Display the curves of loss and accuracy during the training
'''

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

# overfitting is the biggest issue with a small data set.
# this can be seen in these graphs where the training
# accuracy increases linearly over time, where the validation
# stalls around 70-72%. The validation loss reaches its
# minimum after only five epochs and then stalls, whereas
# the training loss keeps decreasing linearly until it
# reaches nearly 0.

# overfitting is caused by having too few samples to learn
# from, rendering you unable to train a model that can 
# generalize to new data.

'''
5.11 Setting up data augmentation configuration via ImageDataGenerator
'''

# # these are just a few of the options
# datagen = ImageDataGenerator(
#     # value in degrees from 0-180, randomly rotates pictures
#     rotation_range=40,
#     # fraction of total width and height within which to
#     # randomly translate pictre vertically or horizontally
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     # randomly applying shearing transformations
#     shear_range=0.2,
#     # randomly zooming inside pictures
#     zoom_range=0.2,
#     # randomly flips half the images horizontally
#     horizontal_flip=True,
#     # strategy for filling in newly created pixels which
#     # can appear after rotation or width/height shift
#     fill_mode='nearest')

# '''
# 5.12 Displaying some randomly augmented training images
# '''

# from keras.preprocessing import image

# fnames = [os.path.join(train_cats_dir, fname) for
#           fname in os.listdir(train_cats_dir)]

# # choose one image to augment
# img_path = fnames[3]

# # reads the image and resizes it
# img = image.load_img(img_path, target_size=(150, 150))

# # converts it to a Numpy array with shape 150, 150, 3
# x = image.img_to_array(img)

# # reshapes to 1, 150, 150, 3
# x = x.reshape((1,) + x.shape)

# # generates batches of randomly transformed images
# # loops forever so you need to break at some point
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break

# plt.show()

# '''
# With data augmentation the network will never see the same image
# twice. But the inputs are still heavily intercorrelated, because
# they come from a small number or original images - you can't
# produce new information, you can only remix existing information.
# As such this may not be enough to complete get rid of overfitting.
# To further fight overfitting, add a Dropout layer to your model,
# right before the densely connected classifier.
# '''

# '''
# 5.13 Defininig a new convnet that includes dropout
# '''

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu',
#                         input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu',))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu',))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu',))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# # new line for Dropout
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])

# '''
# 5.14 Training the convnet using data-augmentation generators
# '''

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,)

# # NOTE! the validation data shouldn't be augmented
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     # target directory
#     train_dir,
#     # resizes all images to 150 x 150
#     target_size=(150, 150),
#     batch_size=32,
#     # because you use binary crossentropy you need
#     # binary labels
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#      validation_dir,
#      target_size=(150, 150),
#      batch_size=32,
#      class_mode='binary')

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=50,)

# '''
# 5.15 Saving the model
# '''

# model.save('cats_and_dogs_small_2.h5')

# '''
# 5.16 Plot the results again
# '''

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()
