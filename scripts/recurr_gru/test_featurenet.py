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

from scripts.recurr_gru.train_gru_feature import feature_net_3D

from skimage.measure import label
from skimage.morphology import watershed
from skimage.feature import peak_local_max


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


def test_gru(X_test, fgbg_gru_weights_file, conv_gru_weights_file):
    run_fgbg_model = feature_net_3D(
        receptive_field=receptive_field,
        dilated=True,
        n_features=2,
        n_frames=frames_per_batch,
        input_shape=tuple(X_test.shape[1:]))
    run_fgbg_model.load_weights(fgbg_gru_weights_file)


    run_watershed_model = feature_net_3D(
        receptive_field=receptive_field,
        dilated=True,
        n_features=distance_bins,
        n_frames=frames_per_batch,
        input_shape=tuple(X_test.shape[1:]))
    run_watershed_model.load_weights(conv_gru_weights_file)


    test_images = run_watershed_model.predict(X_test)[-1]
    test_images_fgbg = run_fgbg_model.predict(X_test)[-1]

    print('watershed transform shape:', test_images.shape)
    print('segmentation mask shape:', test_images_fgbg.shape)

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

    print('watershed argmax shape:', argmax_images.shape)

    # threshold the foreground/background
    # and remove back ground from watershed transform
    threshold = 0.8
    fg_thresh = test_images_fgbg[..., 1] > threshold

    fg_thresh = np.expand_dims(fg_thresh, axis=-1)
    argmax_images_post_fgbg = argmax_images * fg_thresh


    # Label interior predictions

    watershed_images = []
    for i in range(argmax_images_post_fgbg.shape[0]):
        image = fg_thresh[i, ..., 0]
        distance = argmax_images_post_fgbg[i, ..., 0]

        local_maxi = peak_local_max(test_images[i, ..., -1],
                                    min_distance=15, 
                                    exclude_border=False,
                                    indices=False,
                                    labels=image)

        markers = label(local_maxi)
        segments = watershed(-distance, markers, mask=image)
        watershed_images.append(segments)

    watershed_images = np.array(watershed_images)
    watershed_images = np.expand_dims(watershed_images, axis=-1)


    return argmax_images, watershed_images, fg_thresh


# ==============================================================================
# Plot the results
# ==============================================================================

def plot_results(X_test, test_images_fgbg, fg_thresh, 
    argmax_images, argmax_images_post_fgbg, watershed_images, model_name):
    index = np.random.randint(low=0, high=watershed_images.shape[0])
    frame = np.random.randint(low=0, high=watershed_images.shape[1])

    print('Image:', index)
    print('Frame:', frame)

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(X_test[index, frame, ..., 0])
    ax[0].set_title('Source Image')

    ax[1].imshow(test_images_fgbg[index, frame, ..., 1])
    ax[1].set_title('Segmentation Prediction')

    ax[2].imshow(fg_thresh[index, frame, ..., 0], cmap='jet')
    ax[2].set_title('Thresholded Segmentation')

    ax[3].imshow(argmax_images[index, frame, ..., 0], cmap='jet')
    ax[3].set_title('Watershed Transform')

    ax[4].imshow(argmax_images_post_fgbg[index, frame, ..., 0], cmap='jet')
    ax[4].set_title('Watershed Transform w/o Background')

    ax[5].imshow(watershed_images[index, frame, ..., 0], cmap='jet')
    ax[5].set_title('Watershed Segmentation')

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

    if model_name == 'gru':
        fgbg_gru_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_gru_model_name))
        conv_gru_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(conv_gru_model_name))
        test_images, test_images_fgbg = test_gru(X_test, fgbg_gru_weights_file, conv_gru_weights_file)

    else:
        print("Model not supported, please choose gru or lstm")
        sys.exit()

    argmax_images, watershed_images, fg_thresh = post_process(test_images, test_images_fgbg)

    plot_results(X_test, test_images_fgbg, fg_thresh, 
    argmax_images, argmax_images_post_fgbg, watershed_images, model_name)
    get_video_prediction(labeled_images, model_name)


if __name__== "__main__":

    # create directories if they do not exist
    for d in MODEL_DIR:
        try:
            os.makedirs(d)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Set up testing parameters

    conv_gru_model_name = 'conv_gru_featurenet_model'
    fgbg_gru_model_name = 'fgbg_gru_featurenet_model'

    n_epoch = 1  # Number of training epochs
    test_size = .10  # % of data saved as test
    norm_method = 'std'  # data normalization
    receptive_field = 61  # should be adjusted for the scale of the data

    # Transformation settings
    # Transformation settings
    transform = 'watershed'
    distance_bins = 4  # number of distance classes
    erosion_width = 0  # erode edges

    n_features = 4  # (cell-background edge, cell-cell edge, cell interior, background)

    # 3D Settings
    frames_per_batch = 3
    norm_method = 'whole_image'  # data normalization - `whole_image` for 3d conv

    batch_size = 2  # number of images per batch (should be 2 ^ n)
    win = (receptive_field - 1) // 2  # sample window size
    win_z = (frames_per_batch - 1) // 2 # z window size
    balance_classes = True  # sample each class equally
    max_class_samples = 1e7  # max number of samples per class.


    main(sys.argv[1:])

