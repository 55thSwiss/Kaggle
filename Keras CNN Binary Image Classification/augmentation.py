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

# amount of augmented data to create
# split will be 75% training, 18% validation, 7% testing
augmented_data_qty = 1000

# retrieves working directory
cwd = os.getcwd()
# sample data folder
samples_dir = (str(cwd) + (r'\samples'))
# get the number of samples
path, dirs, files = next(os.walk(samples_dir))
samples_count = len(files)
# how many samples in each binary group
binary_samples_count = samples_count / 2

# data subsets
directory_list = ['\\train_t',
                  '\\test_t',
                  '\\validation_t']

# if subset folders don't exist, create them
for d in directory_list:
    if not os.path.exists(str(cwd) + d):
        os.makedirs(str(cwd) + d + '\\0')
        os.makedirs(str(cwd) + d + '\\1')

# for filename in os.listdir(originals_dir):
# rescaling is disabled to allow the images to be viewed
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

for filename in os.listdir(samples_dir):
    # this is a PIL image
    img = load_img(str(samples_dir) + '\\' + str(filename))
    # this is a Numpy array with shape (3, 150, 150)
    x = img_to_array(img)
    # this is a Numpy array with shape (1, 3, 150, 150)
    x = x.reshape((1,) + x.shape)

    # decide destination folder, 0 or 1
    destination = filename.split('_')[0]

    # the .flow() command below generates batches of randomly transformed
    # images and saves the results to save_to_dir
    # training data
    j = 0
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=(str(cwd) + directory_list[0]
                                           + '\\' + destination),
                              save_prefix=str(destination),
                              save_format='jpeg'):
        i += 1
        j += 1
        print(j)
        # check to stop generator from looping indefinitely
        if i > (((augmented_data_qty * 0.75) / 2) / binary_samples_count):
            break
    # testing data
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=(str(cwd) + directory_list[1]
                                           + '\\' + destination),
                              save_prefix=str(destination),
                              save_format='jpeg'):
        i += 1
        j += 1
        print(j)
        if i > (((augmented_data_qty * 0.07) / 2) / binary_samples_count):
            break
    # validation data
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=(str(cwd) + directory_list[2]
                                           + '\\' + destination),
                              save_prefix=str(destination),
                              save_format='jpeg'):
        i += 1
        j += 1
        print(j)
        if i > (((augmented_data_qty * 0.18) / 2) / binary_samples_count):
            break
