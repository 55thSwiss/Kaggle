import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


'''
- rotation_range is a value in degrees (0-180), a range within which to
    randomly rotate pictures
- width_shift and height_shift are ranges (as a fraction of total width or
    height) within which to randomly translate pictures vertically
    or horizontally
- rescale is a value by which we will multiply the data before any other
    processing. Our original images consist in RGB coefficients in the 0-255,
    but such values would be too high for our models to process (given a
    typical learning rate), so we target values between 0 and 1 instead by
    scaling with a 1/255. factor.
- shear_range is for randomly applying shearing transformations
- zoom_range is for randomly zooming inside pictures
- horizontal_flip is for randomly flipping half of the images horizontally --
    relevant when there are no assumptions of horizontal assymetry
    (e.g. real-world pictures).
- fill_mode is the strategy used for filling in newly created pixels, which
    can appear after a rotation or a width/height shift.
'''

cwd = os.getcwd()
path = str(cwd) + (r'\originals\1\\')

# for filename in os.listdir(path):
# rescaling is disabled to allow the images to be viewed
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# this is a PIL image # path + filename
img = load_img(r'C:\Users\Mac2\Desktop\CNN\Chips\originals\1\1_B.jpg')
# this is a Numpy array with shape (3, 150, 150)
x = img_to_array(img)
# this is a Numpy array with shape (1, 3, 150, 150)
x = x.reshape((1,) + x.shape)

# the .flow() command below generates batches of randomly transformed
# images and saves the results to save_to_dir - remember to change prefix
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=(str(cwd) + r'\augmented\train\1'),
                          save_prefix='1', save_format='jpeg'):
    i += 1
    if i > 20:  # change the amount of augmented data you want here
        break  # otherwise the generator would loop indefinitely

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=(str(cwd) + r'\augmented\test\1'),
                          save_prefix='1', save_format='jpeg'):
    i += 1
    if i > 280:  # change the amount of augmented data you want here
        break  # otherwise the generator would loop indefinitely

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=(str(cwd) + r'\augmented\validation\1'),
                          save_prefix='1', save_format='jpeg'):
    i += 1
    if i > 280:  # change the amount of augmented data you want here
        break  # otherwise the generator would loop indefinitely
