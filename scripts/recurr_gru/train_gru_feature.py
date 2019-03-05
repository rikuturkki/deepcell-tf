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
import getopt
import errno

path = sys.path[0]
parentdir = path.replace("scripts/recurr_gru","")
sys.path.insert(0,parentdir) 

import math
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD

from tensorflow.python.keras.utils.data_utils import get_file

import deepcell
from deepcell import losses
from scripts.recurr_gru import image_gen
from deepcell import image_generators
from deepcell import model_zoo
from deepcell.layers import TensorProduct, ReflectionPadding3D, DilatedMaxPool3D

from deepcell.utils import train_utils
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler
from deepcell.training import train_model_conv


from tensorflow.python.keras.layers import MaxPool3D, Conv3DTranspose, UpSampling3D
from scripts.recurr_gru.conv_gru_layer import ConvGRU2D
from tensorflow.python.keras.layers import BatchNormalization, Dropout, LeakyReLU
from tensorflow.python.keras.layers import Conv3D, ZeroPadding3D, ConvLSTM2D, Cropping3D
from tensorflow.python.keras.layers import Input, Add, Concatenate, Flatten
from tensorflow.python.keras.engine.input_layer import InputLayer

from tensorflow.python.keras.models import Model


from tensorflow.python.keras.regularizers import l2
from deepcell.layers import ImageNormalization3D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Softmax

from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib



# Set up file paths
MODEL_DIR = os.path.join(sys.path[0], 'scripts/recurr_gru/models/')
LOG_DIR = os.path.join(sys.path[0], 'scripts/recurr_gru/logs/')


# ==============================================================================
# Models
# ==============================================================================


def feature_net_3D(receptive_field=61,
                      n_frames=5,
                      input_shape=(5, 256, 256, 1),
                      n_features=3,
                      n_channels=1,
                      reg=1e-5,
                      n_conv_filters=64,
                      n_dense_filters=200,
                      VGG_mode=False,
                      init='he_normal',
                      norm_method='std',
                      location=False,
                      dilated=False,
                      padding=False,
                      padding_mode='reflect',
                      multires=False,
                      include_top=True):
    # Create layers list (x) to store all of the layers.
    # We need to use the functional API to enable the multiresolution mode
    x = []

    win = (receptive_field - 1) // 2
    win_z = (n_frames - 1) // 2

    if dilated:
        padding = True

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        time_axis = 2
        row_axis = 3
        col_axis = 4
        if not dilated:
            input_shape = (n_channels, n_frames, receptive_field, receptive_field)
    else:
        channel_axis = -1
        time_axis = 1
        row_axis = 2
        col_axis = 3
        if not dilated:
            input_shape = (n_frames, receptive_field, receptive_field, n_channels)

    x.append(Input(shape=input_shape))
    # x.append(ImageNormalization3D(norm_method=norm_method, filter_size=receptive_field)(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))

    if padding:
        if padding_mode == 'reflect':
            x.append(ReflectionPadding3D(padding=(win_z, win, win))(x[-1]))
        elif padding_mode == 'zero':
            x.append(ZeroPadding3D(padding=(win_z, win, win))([-1]))

    if location:
        x.append(Location3D(in_shape=tuple(x[-1].shape.as_list()[1:]))(x[-1]))
        x.append(Concatenate(axis=channel_axis)([x[-2], x[-1]]))

    if multires:
        layers_to_concat = []

    rf_counter = receptive_field
    block_counter = 0
    d = 1

    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        x.append(Conv3D(n_conv_filters, (1, filter_size, filter_size), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
        x.append(BatchNormalization(axis=channel_axis)(x[-1]))
        x.append(Activation('relu')(x[-1]))

        block_counter += 1
        rf_counter -= filter_size - 1

        if block_counter % 2 == 0:
            if dilated:
                x.append(DilatedMaxPool3D(dilation_rate=(1, d, d), pool_size=(1, 2, 2))(x[-1]))
                d *= 2
            else:
                x.append(MaxPool3D(pool_size=(1, 2, 2))(x[-1]))

            if VGG_mode:
                n_conv_filters *= 2

            rf_counter = rf_counter // 2

            if multires:
                layers_to_concat.append(len(x) - 1)

    if multires:
        c = []
        for l in layers_to_concat:
            output_shape = x[l].get_shape().as_list()
            target_shape = x[-1].get_shape().as_list()
            time_crop = (0, 0)

            row_crop = int(output_shape[row_axis] - target_shape[row_axis])

            if row_crop % 2 == 0:
                row_crop = (row_crop // 2, row_crop // 2)
            else:
                row_crop = (row_crop // 2, row_crop // 2 + 1)

            col_crop = int(output_shape[col_axis] - target_shape[col_axis])

            if col_crop % 2 == 0:
                col_crop = (col_crop // 2, col_crop // 2)
            else:
                col_crop = (col_crop // 2, col_crop // 2 + 1)

            cropping = (time_crop, row_crop, col_crop)

            c.append(Cropping3D(cropping=cropping)(x[l]))
        x.append(Concatenate(axis=channel_axis)(c))

    x.append(Conv3D(n_dense_filters, (1, rf_counter, rf_counter), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(Conv3D(n_dense_filters, (n_frames, 1, 1), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(TensorProduct(n_dense_filters, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                        activation = 'relu', 
                        padding='same', kernel_initializer=init,
                        kernel_regularizer=l2(reg), return_sequences=True)(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                        activation = 'relu', 
                        padding='same', kernel_initializer=init,
                        kernel_regularizer=l2(reg), return_sequences=True)(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))

    x.append(TensorProduct(n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))

    if not dilated:
        x.append(Flatten()(x[-1]))

    if include_top:
        x.append(Softmax(axis=channel_axis)(x[-1]))

    model = Model(inputs=x[0], outputs=x[-1])
    model.summary()

    return model


def feature_net_skip_3D(receptive_field=61,
                           input_shape=(5, 256, 256, 1),
                           fgbg_model=None,
                           last_only=True,
                           n_skips=2,
                           norm_method='std',
                           padding_mode='reflect',
                           **kwargs):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    inputs = Input(shape=input_shape)
    # img = ImageNormalization3D(norm_method=norm_method, filter_size=receptive_field)(inputs)
    img = BatchNormalization(axis=channel_axis)(inputs)

    models = []
    model_outputs = []

    if fgbg_model is not None:
        for layer in fgbg_model.layers:
            layer.trainable = False
        models.append(fgbg_model)
        fgbg_output = fgbg_model(inputs)
        if isinstance(fgbg_output, list):
            fgbg_output = fgbg_output[-1]
        model_outputs.append(fgbg_output)

    for _ in range(n_skips + 1):
        if model_outputs:
            model_input = Concatenate(axis=channel_axis)([img, model_outputs[-1]])
        else:
            model_input = img
        new_input_shape = model_input.get_shape().as_list()[1:]
        models.append(feature_net_3D(receptive_field=receptive_field, input_shape=new_input_shape, norm_method=None, dilated=True, padding=True, padding_mode=padding_mode, **kwargs))
        model_outputs.append(models[-1](model_input))

    if last_only:
        model = Model(inputs=inputs, outputs=model_outputs[-1])
    else:
        if fgbg_model is None:
            model = Model(inputs=inputs, outputs=model_outputs)
        else:
            model = Model(inputs=inputs, outputs=model_outputs[1:])

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
                optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                transform=None,
                log_dir=LOG_DIR,
                model_dir=MODEL_DIR,
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

    print("Training: ", model_name)

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

    print("Input shape of model: ", model.input_shape)
    print("Output shape of model: ", model.output_shape)

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

def create_and_train_fgbg(data_filename, train_dict):
    
    fgbg_model = feature_net_skip_3D(
        receptive_field=receptive_field,
        n_features=2,
        n_frames=frames_per_batch,
        n_skips=n_skips,
        n_conv_filters=32,
        n_dense_filters=128,
        input_shape=tuple(X_test.shape[1:]),
        multires=False,
        last_only=False,
        norm_method=norm_method)

    print("Training fgbg model. \n")

    fgbg_model = train_model_conv(
        model=fgbg_model,
        dataset=data_filename,  # full path to npz file
        model_name=fgbg_gru_model_name,
        log_dir=LOG_DIR,
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

    # Save model
    fgbg_gru_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_gru_model_name))
    fgbg_model.save_weights(fgbg_gru_weights_file)

    return fgbg_model

def create_and_train_gru(data_filename, train_dict):
    conv_gru_model = feature_net_skip_3D(
        receptive_field=receptive_field,
        n_skips=n_skips,
        n_features=4,  # (background edge, interior edge, cell interior, background)
        n_frames=frames_per_batch,
        n_conv_filters=32,
        n_dense_filters=128,
        multires=False,
        last_only=False,
        input_shape=tuple([frames_per_batch] + list(train_dict['X'].shape[2:])),
        norm_method=norm_method)

    print("Training segmentation model. \n")

    conv_gru_model = train_model_conv(
        model=conv_gru_model,
        dataset=data_filename,  # full path to npz file
        model_name=conv_gru_model_name,
        log_dir=LOG_DIR,
        optimizer=optimizer,
        transform=transform,
        dilation_radius=dilation_radius,
        batch_size=batch_size,
        frames_per_batch=frames_per_batch,
        n_epoch=n_epoch,
        model_dir=MODEL_DIR,
        lr_sched=lr_sched,
        rotation_range=180,
        flip=True,
        shear=False,
        zoom_range=(0.8, 1.2))

    # Save model
    conv_gru_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(conv_gru_model_name))
    conv_gru_model.save_weights(conv_gru_weights_file)



# ==============================================================================
# Main loop
# ==============================================================================


def main(argv):
    data_filename = None
    model_name = None
    try:
        opts, args = getopt.getopt(argv,"hf:",["file="])
    except getopt.GetoptError:
        print('train_gru_feature.py -f <full data file path> -m <model name>\n model name is conv or fgbg')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_gru_feature.py -f <full data file path> -m <model name>\n model name is conv or fgbg')
            sys.exit()
        elif opt in ("-f", "--file"):
            data_filename = arg

    if data_filename == None:
        data_filename = 'mousebrain.npz'

    #  Load data
    print("Loading data from " + data_filename)
    train_dict, test_dict = get_data(data_filename, test_size=0.2)

    # Train model and get GPU info
    print("Training GRU")
    print(device_lib.list_local_devices())

    # fgbg_model =  create_and_train_fgbg(data_filename, train_dict)
    create_and_train_gru(data_filename, train_dict)

if __name__== "__main__":

    # create directories if they do not exist
    for d in (MODEL_DIR, LOG_DIR):
        try:
            os.makedirs(d)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Set up training parameters
    conv_gru_model_name = 'conv_gru_featurenet_model'
    fgbg_gru_model_name = 'fgbg_gru_featurenet_model'

    n_epoch = 5  # Number of training epochs
    test_size = .10  # % of data saved as test
    norm_method = 'std'  # data normalization
    receptive_field = 61  # should be adjusted for the scale of the data
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    lr_sched = rate_scheduler(lr=0.01, decay=0.99)

    # FC training settings
    n_skips = 1 # number of skip-connections (only for FC training)
    batch_size = 1  # FC training uses 1 image per batch

    # Transformation settings
    transform = 'deepcell'
    dilation_radius = 1  # change dilation radius for edge dilation
    n_features = 4  # (cell-background edge, cell-cell edge, cell interior, background)

    # 3D Settings
    frames_per_batch = 3
    norm_method = 'whole_image'  # data normalization - `whole_image` for 3d conv

    main(sys.argv[1:])


