import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate, BatchNormalization, Flatten, Input
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from glob import glob
import random
import cv2
import numpy as np
import time


# class to track prograss as for loop is going through an array
class progress_tracker:
    def __init__(self, array):
        array_length = len(array)
        self.one_percent = int(array_length / 100)

    # check if one percent of progress has passed
    def check(self, index):
        if index % self.one_percent == 0:
            temp = str(int(index / self.one_percent))
            print(temp)

    def set_new_array(self, array):
        array_length = len(array)
        self.one_percent = int(array_length / 100)


# class to generate training data from loaded image and mask arrays
class random_line_gen:
    def __init__(self, img_array, mask_array):
        # the arrays to hold training data before being equalized
        ares = []
        ares_nots = []
        self.data = []
        self.blank_line = np.zeros((224), dtype=np.float32)
        progress = progress_tracker(img_array)
        # run through image and mask 
        for image_index, img in enumerate(img_array):
            progress.check(image_index)
            for line_index, line in enumerate(img):
                img_line = img_array[image_index][line_index]
                mask_line = mask_array[image_index][line_index]
                # if any value in the mask is true
                if np.any(mask_line):
                    # normalize line vector 
                    img_normal = img_line / 255
                    # add in line of image and line of mask 
                    ares += [[img_normal, np.array(mask_line, dtype=np.float32)]]
                    # create new lines that add in random data to augment training process 
                    not_mask, not_line, am_mask, am_line = self.new_lines(img_normal, mask_line)
                    ares_nots += [[not_line, not_mask]]
                    ares += [[am_line, am_mask]]
                else:
                    # no value in mask is true so add in blank line
                    ares_nots += [[img_line / 255, np.array(mask_line, dtype=np.float32)]]
        # to make sure both the not and are examples have an equal porportion 
        minimum = min(len(ares), len(ares_nots))
        # shuffle both
        np.random.shuffle(ares)
        np.random.shuffle(ares_nots)
        for i in range(minimum):
            self.data += [ares[i]]
        for i in range(minimum):
            self.data += [ares_nots[i]]
        # shuffle final
        np.random.shuffle(self.data)
        # set the length of the final array 
        self.length = len(self.data)

    # return length of final array 
    def get_length(self):
        return self.length

    # return img and mask lines to the generator
    def get_line(self):
        index = random.randint(0, self.length - 1)
        line = self.data[index]
        img_line = line[0]
        mask_line = line[1]
        return img_line, mask_line

    # creates new lines from random data
    def new_lines(self, img_line, mask_line):
        # create random line
        rand_line = np.random.rand(224, 3)
        # create array copies to add in arandom data 
        are_line = np.copy(img_line)
        are_not_line = np.copy(img_line)
        for index, pixel in enumerate(mask_line):
            if pixel:
                are_not_line[index] = rand_line[index]
            else:
                are_line[index] = rand_line[index]

        return self.blank_line, are_not_line, mask_line, are_line


# data generator to train model
def generator(line_generator, batch_size=32):
    while True:
        batch_input = []
        batch_output = []
        for i in range(batch_size):
            img_line, mask_line = line_generator.get_line()
            batch_input += [img_line]
            batch_output += [mask_line]
        batch_input = np.array(batch_input)
        batch_output = np.array(batch_output)
        yield batch_input, batch_output


# constructing the model 
vertical_input = Input(shape=(224, 3), name='ver_input')
line_feature_layers_vertical = [
    vertical_input,
    Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(224, 3)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(112, 32)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, padding='same', input_shape=(56, 64)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=256, kernel_size=3, padding='same', input_shape=(28, 128)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=512, kernel_size=3, padding='same', input_shape=(14, 256)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=512, kernel_size=3, padding='same', input_shape=(7, 512)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    Dense(512), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
    Dense(512), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
    Dense(512), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
    Dense(512), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
    Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
]
line_features_vertical = tf.keras.models.Sequential(line_feature_layers_vertical)
model = Sequential(line_feature_layers_vertical)
model.compile(
    optimizer='adam',
    loss="mse",
    metrics=['accuracy', 'mae']
)
images = []
print("loading masks ")
masks = np.load("G:/machinelearning/bird/masks/mask.npy")
print("loading images")
full_bird = glob("G:/machinelearning/bird/oriL/*.jpg")
progress = progress_tracker(full_bird)
for i, full in enumerate(full_bird):
    progress.check()
    img = cv2.imread(full)
    images += [img]
print("done")
# create the line generator
line_generator = random_line_gen(images, masks)
# get the length to determine step size
data_length = line_generator.get_length()
print("size of data", data_length)
# determine batch and step sizes 
batch_size = 128
data_size = data_length
steps = int(data_size / batch_size)
data_gen = generator(line_generator, batch_size=128)
model.fit_generator(data_gen, steps_per_epoch=steps, epochs=1, verbose=1)
model.save("E:/machinelearning/saved models/bird_seg_blank_out.h5")
