# ==============================================================================
# Functions for training convolutional GRU
# 
# Heavily borrowed from training.py in deepcell
# ==============================================================================


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import os
import sys
import errno

path = sys.path[0]
parentdir = path.replace("scripts/recurr_gru","")
sys.path.insert(0,parentdir) 


import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD

from tensorflow.python.keras.utils.data_utils import get_file

import deepcell
from deepcell import losses
from deepcell import image_generators
from deepcell import model_zoo
from deepcell.utils import train_utils
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler
from deepcell.training import train_model_conv

from tensorflow.python.keras.layers import MaxPool3D
from scripts.recurr_gru.conv_gru_layer import ConvGRU2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv3D
from tensorflow.python.keras.regularizers import l2
from deepcell.layers import ImageNormalization3D
from tensorflow.python.keras.models import Sequential


from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ==============================================================================
# Set up file paths
# ==============================================================================

MODEL_DIR = os.path.join('models')
LOG_DIR = os.path.join('logs')

# create directories if they do not exist
for d in (MODEL_DIR, LOG_DIR):
    try:
        os.makedirs(d)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


# ==============================================================================
# Load data
# ==============================================================================


def load_data(file_name, test_size=.2, seed=0):
    """Loads the dataset.

    # Args:
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        test_size: fraction of data to reserve as test data
        seed: the seed for randomly shuffling the dataset

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    training_data = np.load(file_name)

    X = training_data['X']
    y = training_data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    train_dict = {
        'X': X_train,
        'y': y_train
    }

    test_dict = {
        'X': X_test,
        'y': y_test
    }

    return train_dict, test_dict 

data_filename = 'nuclear_movie_hela0-7_same.npz'
train_dict, test_dict = get_data(data_filename, test_size=0.2)

# ==============================================================================
# Set up training parameters
# ==============================================================================


conv_gru_model_name = 'conv_gru_model'
fgbg_model_name = 'conv_fgbg_model'


n_epoch = 10  # Number of training epochs
test_size = .10  # % of data saved as test
receptive_field = 61  # should be adjusted for the scale of the data

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

lr_sched = rate_scheduler(lr=0.01, decay=0.99)


# Transformation settings
transform = None
n_features = 4  # (cell-background edge, cell-cell edge, cell interior, background)

# 3D Settings
frames_per_batch = 3
norm_method = 'whole_image'  # data normalization - `whole_image` for 3d conv


# ==============================================================================
# Get model
# ==============================================================================


def feature_net_3D(receptive_field=61,
                      n_frames=5,
                      input_shape=(5, 256, 256, 1),
                      n_features=3,
                      n_channels=1,
                      reg=1e-5,
                      n_conv_filters=64,
                      n_dense_filters=200,
                      init='he_normal',
                      norm_method='std',
                      include_top=True):
    # Create layers list (x) to store all of the layers.
    # We need to use the functional API to enable the multiresolution mode

    win = (receptive_field - 1) // 2
    win_z = (n_frames - 1) // 2


    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        time_axis = 2
        row_axis = 3
        col_axis = 4
        input_shape = (n_channels, n_frames, receptive_field, receptive_field)
    else:
        channel_axis = -1
        time_axis = 1
        row_axis = 2
        col_axis = 3
        input_shape = (n_frames, receptive_field, receptive_field, n_channels)

    model = Sequential()
    model.add(ImageNormalization3D(norm_method=norm_method, 
        filter_size=receptive_field, input_shape=input_shape))
    
    rf_counter = receptive_field
    block_counter = 0
    d = 1

    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        model.add(Conv3D(n_conv_filters, kernel_size=(1, filter_size, filter_size), 
            dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', 
            kernel_regularizer=l2(reg), activation='relu'))
        model.add(BatchNormalization(axis=channel_axis))

        block_counter += 1
        rf_counter -= filter_size - 1

        if block_counter % 2 == 0:
            model.add(MaxPool3D(pool_size=(1, 2, 2)))

        rf_counter = rf_counter // 2


    model.add(ConvGRU2D(filters=n_dense_filters, kernel_size=(filter_size, filter_size), 
                    kernel_regularizer=l2(reg), padding='valid', return_sequences=True))
    model.add(BatchNormalization(axis=channel_axis))


    model.add(ConvGRU2D(filters=n_dense_filters, kernel_size=(filter_size, filter_size), 
                    kernel_regularizer=l2(reg), padding='valid', return_sequences=True))
    model.add(BatchNormalization(axis=channel_axis))


    model.add(ConvGRU2D(filters=n_dense_filters, kernel_size=(filter_size, filter_size), 
                    kernel_regularizer=l2(reg), padding='valid', return_sequences=True))
    model.add(BatchNormalization(axis=channel_axis))


    model.add(ConvGRU2D(filters=n_dense_filters, kernel_size=(filter_size, filter_size), 
                    kernel_regularizer=l2(reg), padding='valid', return_sequences=True))
    model.add(BatchNormalization(axis=channel_axis))


    model.add(Conv3D(filters=n_dense_filters, kernel_size=(1, filter_size, filter_size), 
                    kernel_initializer=init, 
                    kernel_regularizer=l2(reg), activation='relu',
                   padding='same', data_format='channels_last'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return model


# ==============================================================================
# Train model
# ==============================================================================

def train_model(model,
                data_filename,
                expt='',
                test_size=.1,
                n_epoch=10,
                batch_size=1,
                num_gpus=None,
                frames_per_batch=5,
                transform=None,
                optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                log_dir='/data/tensorboard_logs',
                model_dir='/data/models',
                model_name=None,
                focal=False,
                gamma=0.5,
                lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                rotation_range=0,
                flip=True,
                shear=0,
                zoom_range=0,
                **kwargs):
    is_channels_first = K.image_data_format() == 'channels_first'

    if model_name is None:
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = '{}_{}_{}'.format(todays_date, data_name, expt)
    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))


    train_dict, test_dict = get_data(data_filename, test_size=test_size)


    n_classes = model.layers[-1].output_shape[1 if is_channels_first else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        if isinstance(transform, str) and transform.lower() == 'disc':
            return losses.discriminative_instance_loss(y_true, y_pred)
        if focal:
            return losses.weighted_focal_loss(
                y_true, y_pred, gamma=gamma, n_classes=n_classes)
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    if num_gpus >= 2:
        batch_size = batch_size * num_gpus
        model = train_utils.MultiGpuModel(model, num_gpus)

    print('Training on {} GPUs'.format(num_gpus))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    if isinstance(model.output_shape, list):
        skip = len(model.output_shape) - 1
    else:
        skip = None

    if train_dict['X'].ndim == 5:
        DataGenerator = image_generators.MovieDataGenerator
    else:
        raise ValueError('Expected `X` to have ndim 5. Got',
                         train_dict['X'].ndim)

    if num_gpus >= 2:
        # Each GPU must have at least one validation example
        if test_dict['y'].shape[0] < num_gpus:
            raise ValueError('Not enough validation data for {} GPUs. '
                             'Received {} validation sample.'.format(
                                 test_dict['y'].shape[0], num_gpus))

        # When using multiple GPUs and skip_connections,
        # the training data must be evenly distributed across all GPUs
        num_train = train_dict['y'].shape[0]
        nb_samples = num_train - num_train % batch_size
        if nb_samples:
            train_dict['y']  = train_dict['y'][:nb_samples]
            train_dict['X'] = train_dict['X'][:nb_samples]

    # this will do preprocessing and realtime data augmentation
    datagen = DataGenerator(
        rotation_range=rotation_range,
        shear_range=shear,
        zoom_range=zoom_range,
        horizontal_flip=flip,
        vertical_flip=flip)

    datagen_val = DataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    if train_dict['X'].ndim == 5:
        train_data = datagen_val.flow(
            train_dict,
            skip=skip,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs,
            frames_per_batch=frames_per_batch)

        val_data = datagen_val.flow(
            test_dict,
            skip=skip,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=kwargs,
            frames_per_batch=frames_per_batch)
    else:
        raise ValueError('Expected `X` to have ndim 5. Got',
                         train_dict['X'].ndim)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=[
            callbacks.LearningRateScheduler(lr_sched),
            callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=num_gpus >= 2),
            callbacks.TensorBoard(log_dir=os.path.join(log_dir, model_name))
        ])

    model.save_weights(model_path)
    np.savez(loss_path, loss_history=loss_history.history)

    return model

# ==============================================================================
# Create and train foreground/background separation model
# ==============================================================================

fgbg_model = model_zoo.bn_feature_net_3D(
    n_features=2,  # segmentation mask (is_cell, is_not_cell)
    n_frames=frames_per_batch,
    n_conv_filters=32,
    n_dense_filters=128,
    input_shape=tuple([frames_per_batch] + list(train_dict['X'].shape[2:])),
    multires=False,
    last_only=False,
    norm_method=norm_method)

fgbg_model = train_model(
    model=fgbg_model,
    data_filename=data_filename,  # full path to npz file
    model_name=fgbg_model_name,
    transform='fgbg',
    optimizer=optimizer,
    batch_size=batch_size,
    frames_per_batch=frames_per_batch,
    n_epoch=n_epoch,
    model_dir=MODEL_DIR,
    lr_sched=lr_sched,
    rotation_range=180,
    flip=True,
    shear=False,
    zoom_range=(0.8, 1.2))

# ==============================================================================
# Create a segmentation model
# ==============================================================================

conv_gru_model = feature_net_3D(
    receptive_field=receptive_field,
    n_features=4, 
    n_frames=frames_per_batch,
    n_conv_filters=32,
    n_dense_filters=128,
    input_shape=tuple([frames_per_batch] + list(train_dict['X'].shape[2:])),
    norm_method=norm_method)

conv_gru_model = train_model(
    model=conv_gru_model,
    data_filename = data_filename,
    model_name=conv_gru_model_name,
    optimizer=optimizer,
    transform=transform,
    frames_per_batch=frames_per_batch,
    n_epoch=n_epoch,
    model_dir=MODEL_DIR,
    lr_sched=lr_sched,
    rotation_range=180,
    flip=True,
    shear=False,
    zoom_range=(0.8, 1.2))

# ==============================================================================
# Save weights of trained models
# ==============================================================================

fgbg_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_model_name))
fgbg_model.save_weights(fgbg_weights_file)

conv_gru_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(conv_gru_model_name))
conv_gru_model.save_weights(conv_gru_weights_file)



