# ==============================================================================
# Functions for testing convolutional GRU
# 
# Referenced from Interior-Edge Segmentation 3D Fully Convolutional.ipynb
# ==============================================================================


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import os
import sys
import errno
import getopt

path = sys.path[0]
parentdir = path.replace("scripts/recurr_gru","")
sys.path.insert(0,parentdir) 

import deepcell
from deepcell import losses
from deepcell import image_generators
from deepcell import model_zoo
from deepcell.utils.data_utils import get_data
from deepcell.utils.plot_utils import get_js_video

from scripts.recurr_gru.train_GRU import feature_net_GRU
from scripts.recurr_gru.train_LSTM import feature_net_LSTM

import numpy as np
from skimage.measure import label
from skimage import morphology
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import pdb

from tensorflow.python.client import device_lib

MODEL_DIR = os.path.join(sys.path[0], 'scripts/recurr_gru/models')


# ==============================================================================
# Initialize new model
# ==============================================================================


def test_lstm(X_test, fgbg_lstm_weights_file, conv_lstm_weights_file):
    run_fgbg_model = feature_net_LSTM(
        input_shape=tuple(X_test.shape[1:]),
        receptive_field=receptive_field,
        n_features=2, 
        n_frames=frames_per_batch,
        n_conv_filters=32,
        n_dense_filters=128,
        norm_method=norm_method)
    run_fgbg_model.load_weights(fgbg_lstm_weights_file)


    run_conv_model = feature_net_LSTM(
        input_shape=tuple(X_test.shape[1:]),
        receptive_field=receptive_field,
        n_features=4,  # (background edge, interior edge, cell interior, background)
        n_frames=frames_per_batch,
        n_conv_filters=32,
        n_dense_filters=128,
        norm_method=norm_method)
    run_conv_model.load_weights(conv_lstm_weights_file)


    test_images = run_conv_model.predict(X_test)[-1]
    test_images_fgbg = run_fgbg_model.predict(X_test)[-1]

    print('edge/interior prediction shape:', test_images.shape)
    print('fgbg mask shape:', test_images_fgbg.shape)

    return test_images, test_images_fgbg


def test_gru(X_test, fgbg_gru_weights_file, conv_gru_weights_file):
    run_fgbg_model = feature_net_GRU(
        input_shape=tuple(X_test.shape[1:]),
        receptive_field=receptive_field,
        n_features=2, 
        n_frames=frames_per_batch,
        n_conv_filters=32,
        n_dense_filters=128,
        norm_method=norm_method)
    run_fgbg_model.load_weights(fgbg_gru_weights_file)


    run_conv_model = feature_net_GRU(
        input_shape=tuple(X_test.shape[1:]),
        receptive_field=receptive_field,
        n_features=4,  # (background edge, interior edge, cell interior, background)
        n_frames=frames_per_batch,
        n_conv_filters=32,
        n_dense_filters=128,
        norm_method=norm_method)
    run_conv_model.load_weights(conv_gru_weights_file)


    test_images = run_conv_model.predict(X_test)[-1]
    test_images_fgbg = run_fgbg_model.predict(X_test)[-1]

    print('edge/interior prediction shape:', test_images.shape)
    print('fgbg mask shape:', test_images_fgbg.shape)

    return test_images, test_images_fgbg

# ==============================================================================
# Post processing
# ==============================================================================

def post_process(test_images, test_images_fgbg):
    argmax_images = []
    for i in range(test_images.shape[0]):
        max_image = np.argmax(test_images[i], axis=-1)
        argmax_images.append(max_image)
    argmax_images = np.array(argmax_images)
    argmax_images = np.expand_dims(argmax_images, axis=-1)

    print('argmax shape:', argmax_images.shape)

    # threshold the foreground/background
    # and remove back ground from edge transform
    threshold = 0.9

    fg_thresh = test_images_fgbg[..., 1] > threshold
    fg_thresh = np.expand_dims(fg_thresh, axis=-1)

    test_images_post_fgbg = test_images * fg_thresh


    # Label interior predictions

    labeled_images = []
    for i in range(test_images_post_fgbg.shape[0]):
        interior = test_images_post_fgbg[i, ..., 2] > .2
        labeled_image = label(interior)
        labeled_image = morphology.remove_small_objects(
            labeled_image, min_size=50, connectivity=1)
        labeled_images.append(labeled_image)
    labeled_images = np.array(labeled_images)
    labeled_images = np.expand_dims(labeled_images, axis=-1)

    print('labeled_images shape:', labeled_images.shape)

    return labeled_images


# ==============================================================================
# Plot the results
# ==============================================================================

def plot_results(labeled_images, model_name):
    index = np.random.randint(low=0, high=labeled_images.shape[0])
    print('Image number:', index)

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(X_test[0,index, ..., 0])
    ax[0].set_title('Source Image')

    ax[1].imshow(test_images_fgbg[index, ..., 1])
    ax[1].set_title('Segmentation Prediction')

    ax[2].imshow(fg_thresh[index, ..., 0], cmap='jet')
    ax[2].set_title('FGBG Threshold {}%'.format(threshold * 100))

    ax[3].imshow(test_images[index, ..., 0] + test_images[index, ..., 1], cmap='jet')
    ax[3].set_title('Edge Prediction')

    ax[4].imshow(test_images[index, ..., 2], cmap='jet')
    ax[4].set_title('Interior Prediction')

    ax[5].imshow(labeled_images[index, ..., 0], cmap='jet')
    ax[5].set_title('Instance Segmentation')

    fig.tight_layout()
    plt.show()
    plt.savefig(model_name + '_predictions.png')


def get_video(images, batch=0, channel=0, cmap='jet'):
    """Create a JavaScript video as HTML for visualizing 3D data as a movie"""
    fig = plt.figure()

    ims = []
    for i in range(images.shape[0]):
        im = plt.imshow(images[i, :, :, channel], animated=True, cmap=cmap)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=150, repeat_delay=1000)
    plt.close()
    return ani


def get_video_prediction(labeled_images, model_name):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    vid = get_video(labeled_images, batch=0)
    vid.save(model_name + '_predictions.mp4', writer=writer)
    print("Done predictions video.")
    return
# ==============================================================================
# Main loop
# ==============================================================================


def main(argv):
    data_filename = None
    model_name = None
    try:
        opts, args = getopt.getopt(argv,"hf:m:",["file=","model="])
    except getopt.GetoptError:
        print('test.py -f <full data file path> -m <model name>\n model name is gru or lstm')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -f <full data file path> -m <model name>\n model name is gru or lstm')
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
    X_test, y_test = test_dict['X'][:4], test_dict['y'][:4]


    # Train model and get GPU info
    print("Testing " + model_name)

    if model_name == 'lstm':
        fgbg_lstm_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_lstm_model_name))
        conv_lstm_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(conv_lstm_model_name))
        test_images, test_images_fgbg = test_lstm(X_test, fgbg_lstm_weights_file, conv_lstm_weights_file)

    elif model_name == 'gru':
        fgbg_gru_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_gru_model_name))
        conv_gru_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(conv_gru_model_name))
        test_images, test_images_fgbg = test_gru(X_test, fgbg_gru_weights_file, conv_gru_weights_file)

    else:
        print("Model not supported, please choose fgbg or conv-gru")
        sys.exit()

    labeled_images = post_process(test_images, test_images_fgbg)

    plot_results(labeled_images, model_name)
    get_video_prediction(labeled_images, model_name)


if __name__== "__main__":

    # create directories if they do not exist
    for d in MODEL_DIR:
        try:
            os.makedirs(d)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Set up training parameters
    conv_lstm_model_name = 'conv_lstm_model'
    fgbg_lstm_model_name = 'lstm_fgbg_model'

    conv_gru_model_name = 'conv_gru_model'
    fgbg_gru_model_name = 'fgbg_gru_model'

    n_epoch = 10  # Number of training epochs
    test_size = .10  # % of data saved as test
    receptive_field = 61  # should be adjusted for the scale of the data

    batch_size = 1  # FC training uses 1 image per batch

    # Transformation settings
    transform = None
    n_features = 4  # (cell-background edge, cell-cell edge, cell interior, background)

    # 3D Settings
    frames_per_batch = 3
    norm_method = 'whole_image'  # data normalization - `whole_image` for 3d conv

    main(sys.argv[1:])

