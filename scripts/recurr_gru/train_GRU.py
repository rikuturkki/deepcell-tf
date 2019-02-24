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


import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD, RMSprop

from tensorflow.python.keras.utils.data_utils import get_file

import deepcell
from deepcell import losses
from scripts.recurr_gru import image_gen
from deepcell import image_generators
from deepcell import model_zoo
from deepcell.layers import TensorProduct
from deepcell.utils import train_utils
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler
from deepcell.training import train_model_conv

from tensorflow.python.keras.layers import MaxPool3D, Conv3DTranspose, UpSampling3D
from scripts.recurr_gru.conv_gru_layer import ConvGRU2D
from tensorflow.python.keras.layers import BatchNormalization, Dropout
from tensorflow.python.keras.layers import Conv3D, ZeroPadding3D, ConvLSTM2D
from tensorflow.python.keras.layers import Input, Add, Concatenate
from tensorflow.python.keras.engine.input_layer import InputLayer

from tensorflow.python.keras.models import Model


from tensorflow.python.keras.regularizers import l2
from deepcell.layers import ImageNormalization3D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Softmax

from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib



# Set up file paths
MODEL_DIR = os.path.join(sys.path[0], 'scripts/recurr_gru/models')
LOG_DIR = os.path.join(sys.path[0], 'scripts/recurr_gru/logs')


# ==============================================================================
# Models
# ==============================================================================


def feature_net_GRU(input_shape,
                    receptive_field=61,
                    n_frames=5,
                    n_features=3,
                    n_channels=1,
                    reg=1e-5,
                    n_conv_filters=40,
                    n_dense_filters=200,
                    init='he_normal',
                    norm_method='std',
                    include_top=True):
    # Create layers list (x) to store all of the layers.
    
    win = (receptive_field - 1) // 2
    win_z = (n_frames - 1) // 2


    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        time_axis = 2
        row_axis = 3
        col_axis = 4
    else:
        channel_axis = -1
        time_axis = 1
        row_axis = 2
        col_axis = 3

    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))

    rf_counter = receptive_field
    block_counter = 0
    d = 1
    
    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        model.add(ConvGRU2D(n_conv_filters, kernel_size=(filter_size, filter_size), 
            padding='same', 
            kernel_initializer=init,
            kernel_regularizer=l2(reg), 
            activation='relu', 
            return_sequences=True))
        model.add(BatchNormalization(axis=channel_axis))

        block_counter += 1
        rf_counter -= filter_size - 1

        rf_counter = rf_counter // 2
    

    model.add(ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer=init,
                       kernel_regularizer=l2(reg), return_sequences=True))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer=init,
                       kernel_regularizer=l2(reg),
                       return_sequences=True))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(TensorProduct(n_dense_filters, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(TensorProduct(n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    model.summary()
    return model

def feature_net_skip_GRU(input_shape,
                        receptive_field=61,
                        n_frames=5,
                        n_features=3,
                        n_channels=1,
                        reg=1e-5,
                        n_conv_filters=40,
                        n_dense_filters=200,
                        init='he_normal',
                        norm_method='std',
                        include_top=True):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    inputs = Input(shape=input_shape)
    conv1 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(inputs)
    conv1 = BatchNormalization(axis=channel_axis)(conv1)
    conv1 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv1)
    norm1 = BatchNormalization(axis=channel_axis)(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(norm1)

    conv2 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(pool1)
    conv2 = BatchNormalization(axis=channel_axis)(conv2)
    conv2 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv2)
    norm2 = BatchNormalization(axis=channel_axis)(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(norm2)

    conv3 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(pool2)
    conv3 = BatchNormalization(axis=channel_axis)(conv3)
    conv3 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv3)
    norm3 = BatchNormalization(axis=channel_axis)(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(norm3)


    conv4 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(pool3)
    conv4 = BatchNormalization(axis=channel_axis)(conv4)
    conv4 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv4)
    norm4 = BatchNormalization(axis=channel_axis)(conv4)
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(norm4)



    conv5 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(pool4)
    conv5 = BatchNormalization(axis=channel_axis)(conv5)  
    conv5 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv5)
    norm5 = BatchNormalization(axis=channel_axis)(conv5)


    # up1 = UpSampling3D(size=(1, 2, 2))(norm5)

    up1 = Conv3DTranspose(filters=n_conv_filters, kernel_size=(1, 3, 3),
                        strides=(1, 2, 2), padding='same')(norm5)

    joinedTensor1 = Concatenate(axis=channel_axis)([norm4, up1])

    conv6 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(joinedTensor1)
    conv6 = BatchNormalization(axis=channel_axis)(conv6)
    conv6 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv6)
    norm6 = BatchNormalization(axis=channel_axis)(conv6)

    # up2 = UpSampling3D(size=(1, 2, 2))(norm6)
    up2 = Conv3DTranspose(filters=n_conv_filters, kernel_size=(1, 3,3),
                        strides=(1, 2, 2), padding='same')(norm6)


    joinedTensor2 = Concatenate(axis=channel_axis)([norm3, up2])

    conv7 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(joinedTensor2)
    conv7 = BatchNormalization(axis=channel_axis)(conv7)
    conv7 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv7)
    norm7 = BatchNormalization(axis=channel_axis)(conv7)
    
    # up3 = UpSampling3D(size=(1, 2, 2))(norm7)

    up3 = Conv3DTranspose(filters=n_conv_filters, kernel_size=(1, 3,3),
                        strides=(1, 2, 2), padding='same')(norm7)

    joinedTensor3 = Concatenate(axis=channel_axis)([norm2, up3])

    conv8 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    # activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(joinedTensor3)
    conv8 = BatchNormalization(axis=channel_axis)(conv8)
    conv8 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv8)
    norm8 = BatchNormalization(axis=channel_axis)(conv8)
    
    # up4 = UpSampling3D(size=(1, 2, 2))(norm8)

    up4 = Conv3DTranspose(filters=n_conv_filters, kernel_size=(1, 3,3),
                        strides=(1, 2, 2), padding='same')(norm8)

    joinedTensor4 = Concatenate(axis=channel_axis)([norm1, up4])

    conv9 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(joinedTensor4)
    conv9 = BatchNormalization(axis=channel_axis)(conv9)
    conv9 = ConvGRU2D(filters=n_conv_filters, kernel_size=(3, 3),
                    activation = 'relu', 
                    padding='same', kernel_initializer=init,
                    kernel_regularizer=l2(reg), return_sequences=True)(conv9)
    conv9 = BatchNormalization(axis=channel_axis)(conv9)

    # y1 = TensorProduct(n_dense_filters, kernel_initializer=init,
    #                     activation='relu',  kernel_regularizer=l2(reg))(conv9)
    # y1 = BatchNormalization(axis=channel_axis)(y1)
    output = TensorProduct(n_features, kernel_initializer=init, 
                        activation='sigmoid', kernel_regularizer=l2(reg))(conv9)

    model = Model(inputs,output)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
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
    
    fgbg_model = feature_net_skip_GRU(
        input_shape=tuple([frames_per_batch] + list(train_dict['X'].shape[2:])),
        n_features=2,  # segmentation mask (is_cell, is_not_cell)
        n_frames=frames_per_batch,
        n_conv_filters=32,
        n_dense_filters=128,
        norm_method=norm_method)

    # print(fgbg_model.summary())

    fgbg_model = train_model(
        model=fgbg_model,
        data_filename=data_filename,  # full path to npz file
        model_name=fgbg_gru_model_name,
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


# ==============================================================================
# Create a segmentation model
# ==============================================================================

def create_and_train_conv_gru(data_filename, train_dict):
    conv_gru_model = feature_net_skip_GRU(
        input_shape=tuple([frames_per_batch] + list(train_dict['X'].shape[2:])),
        receptive_field=receptive_field,
        n_features=4, 
        n_frames=frames_per_batch,
        n_conv_filters=32,
        n_dense_filters=128,
        norm_method=norm_method)

    # print(conv_gru_model.summary())

    conv_gru_model = train_model(
        model=conv_gru_model,
        data_filename = data_filename,
        model_name=conv_gru_model_name,
        optimizer=optimizer,
        transform='deepcell',
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
        opts, args = getopt.getopt(argv,"hf:m:",["file=","model="])
    except getopt.GetoptError:
        print('train_GRU.py -f <full data file path> -m <model name>\n model name is conv or fgbg')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_GRU.py -f <full data file path> -m <model name>\n model name is conv or fgbg')
            sys.exit()
        elif opt in ("-f", "--file"):
            data_filename = arg
        elif opt in ("-m", "--model"):
            model_name = arg

    if data_filename == None:
        data_filename = 'mousebrain.npz'

    #  Load data
    print("Loading data from " + data_filename)
    train_dict, test_dict = get_data(data_filename, test_size=0.2)

    # Train model and get GPU info
    print("Training " + model_name)
    print(device_lib.list_local_devices())

    if model_name == 'fgbg':
        create_and_train_fgbg(data_filename, train_dict)
    elif model_name == 'conv':
        create_and_train_conv_gru(data_filename, train_dict)
    else:
        print("Model not supported, please choose fgbg or conv")
        sys.exit()

  
if __name__== "__main__":

    # create directories if they do not exist
    for d in (MODEL_DIR, LOG_DIR):
        try:
            os.makedirs(d)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Set up training parameters
    conv_gru_model_name = 'conv_gru_model'
    fgbg_gru_model_name = 'fgbg_gru_model'

    n_epoch = 5  # Number of training epochs
    test_size = .10  # % of data saved as test
    receptive_field = 61  # should be adjusted for the scale of the data

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = RMSprop()

    lr_sched = rate_scheduler(lr=0.01, decay=0.99)
    batch_size = 1  # FC training uses 1 image per batch

    # Transformation settings
    transform = None
    n_features = 4  # (cell-background edge, cell-cell edge, cell interior, background)

    # 3D Settings
    frames_per_batch = 3
    norm_method = 'whole_image'  # data normalization - `whole_image` for 3d conv

    main(sys.argv[1:])


