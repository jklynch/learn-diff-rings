import itertools

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import Sequence

def generate_diffraction_rings(center_x, center_y, radius_list):
    width = 100
    height = 100
    size = (width, height)
    image = Image.new(mode="F", size=size)
    artist = ImageDraw.ImageDraw(image)

    for radius in radius_list:
        artist.ellipse(
            (center_x - radius/2, center_y - radius/2, center_x + radius/2, center_y + radius/2),
            width=1,
            outline=255
        )

    return image

fig, axs = plt.subplots(nrows=2, ncols=2)
for r, c in itertools.product(range(2), range(2)):
    center_x, center_y = np.random.randint(low=20, high=80, size=2)
    ring_count = np.random.randint(low=1, high=5)
    radius_list = sorted(np.random.permutation(tuple(range(20, 71)))[:ring_count])
    print('center: ({}, {}) radius_list: {}'.format(center_x, center_y, radius_list))

    im = generate_diffraction_rings(center_x=center_x, center_y=center_y, radius_list=radius_list)
    axs[r][c].imshow(im)

def preprocess_input(x):
    # convert the data to the right type
    x = x.astype('float32')
    x /= 255
    ##print('x shape:', x.shape)

    #fig, axs = plt.subplots(nrows=1, ncols=2)
    #axs[0].set_title('x_train')
    #axs[0].hist(x_train.flatten())
    #plt.show()

    return x

def generate_data(sample_count):
    # create training and testing data
    # input image dimensions
    img_x, img_y = 100, 100

    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    x = np.zeros((sample_count, img_x, img_y, 1))
    input_shape = (img_x, img_y, 1)

    ##y = np.zeros((sample_count, 1+1+10))
    y = np.zeros((sample_count, 5))

    for i in range(sample_count):
        center_x, center_y = np.random.randint(low=20, high=80, size=2)
        ring_count = np.random.randint(low=1, high=10)
        radius_list = sorted(np.random.permutation(tuple(range(20, 71)))[:ring_count])
        x[i, :, :, 0] = generate_diffraction_rings(
            center_x=center_x, center_y=center_y, radius_list=radius_list
        )
        y[i, 0] = center_x / 100
        y[i, 1] = center_y / 100
        ##y[i, 2:len(radius_list)+2] = radius_list
        y[i, 2] = radius_list[0] / 100
        y[i, 3] = radius_list[-1] / 100
        y[i, 4] = len(radius_list)

    return preprocess_input(x), y

class DiffractionRingSequence(Sequence):
    """
        Generate training and validation data.
    """
    def __init__(self, batch_count_per_epoch, batch_size):
        self.batch_count_per_epoch = batch_count_per_epoch
        self.batch_size = batch_size
        #self.sample_lengths = sample_lengths

    def __len__(self):
        """
            Return the number of batches per epoch.
        """
        return int(self.batch_count_per_epoch)

    def __getitem__(self, index):
        """
            Return one batch of data.
        """
        #this_sample_length = np.random.choice(self.sample_lengths)
        X, y = generate_data(sample_count=self.batch_size)

        return X, y

    def on_epoch_end(self):
        """

        """
        pass

def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(
        32,
        kernel_size=(5, 5),
        activation='relu', kernel_initializer='he_uniform', padding='same',
        input_shape=input_shape
    ))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(output_shape, activation='relu', kernel_initializer='he_uniform'))

    model.compile(
        loss='mean_absolute_error',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def summarize_diagnostics(history):
    # plot loss
    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].set_title('MSE Loss')
    axs[0].plot(history.history['loss'], color='blue', label='train')
    axs[0].plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    axs[1].set_title('Accuracy')
    axs[1].plot(history.history['acc'], color='blue', label='train')
    axs[1].plot(history.history['val_acc'], color='orange', label='test')
    plt.show()
    # save plot to file
    #filename = sys.argv[0].split('/')[-1]
    plt.savefig('learn_plot.')
    plt.close()

def train_model():

    x, y = generate_data(sample_count=1000)
    # prepare pixel data
    ##x_train, x_test = preprocess_input(x_train, x_test)
    # define model
    # x_train.shape looks like (1000, 100, 100, 1)
    print(y.shape)
    model = build_model(input_shape=x.shape[1:], output_shape=y.shape[1])
    model.summary()
    # fit model
    ##history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=1)
    history = model.fit_generator(
        generator=DiffractionRingSequence(
            batch_count_per_epoch=100,
            batch_size=64
        ),
        validation_data=DiffractionRingSequence(
            batch_count_per_epoch=1,
            batch_size=1000
        ),
        epochs=10
    )
    # evaluate model
    _, acc = model.evaluate(x, y, verbose=0)
    print(f'accuracy: {acc * 100:.3f}')
    # learning curves
    summarize_diagnostics(history)

    model.save('diffraction_ring_model.h5')

    return model

train_model()
