# full flowthrough with DeepWatershed instance segmentation

import os
import errno
import argparse

import numpy as np
import skimage.io
import skimage.external.tifffile as tiff
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import backend as K
from scipy import stats

from deepcell import get_image_sizes
from deepcell import make_training_data
from deepcell import bn_feature_net_31x31
from deepcell import dilated_bn_feature_net_31x31
from deepcell import train_model_watershed
from deepcell import train_model_watershed_sample
from deepcell import bn_dense_feature_net
from deepcell import rate_scheduler
from deepcell import train_model_disc, train_model_conv, train_model_sample
from deepcell import run_models_on_directory
from deepcell import export_model
from deepcell import get_data

# data options
DATA_OUTPUT_MODE = 'sample'
BORDER_MODE = 'valid' if DATA_OUTPUT_MODE == 'sample' else 'same'
RESIZE = True
RESHAPE_SIZE = 2048
N_EPOCHS = 40
WINDOW_SIZE = (15,15)
BATCH_SIZE = 64
MAX_TRAIN = 1e8
BINS = 4

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'

PREFIX_SEG = 'tissues/mibi/samir'
PREFIX_CLASS = 'tissues/mibi/mibi_full'
PREFX_SAVE = 'tissues/mibi/pipeline'

FG_BG_DATA_FILE = 'mibi_pipe_wshedFB_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)
WATERSHED_DATA_FILE = 'mibi_pipe_wshed_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)
CONV_DATA_FILE = 'mibi_pipe_wshedconv_{}_{}'.format(K.image_data_format(), 'conv')
CLASS_DATA_FILE = 'mibi_pipe_class_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)

#'2018-07-13_mibi_watershedFB_channels_last_sample_fgbg_0.h5'

MODEL_FGBG = ''
MODEL_WSHED = ''
MODEL_CLASS = ''

RUN_DIR = 'set1'

TRAIN_DIR_SAMPLE = ['set1', 'set2']
TRAIN_DIR_CLASS_RANGE = range(1, 39+1)

NUM_FEATURES_IN_SEG = 2
NUM_FEATURES_OUT_SEG = 3
NUM_FEATURES_CLASS = 17

CHANNELS_SEG = ['dsDNA', 'Ca', 'H3K27me3', 'H3K9ac', 'Ta']  #Add P?
CHANNELS_CLASS = ['dsDNA', 'Ca', 'H3K27me3', 'H3K9ac', 'Ta', 'FoxP3.', 'CD4.', 'CD16.', 'EGFR.', 'CD68.', 'CD8.', 'CD3.',
                 'Keratin17.', 'CD20.', 'p53.', 'catenin.', 'HLA-DR.', 'CD45.', 'Pan-Keratin.', 'MPO.',
                 'Keratin6.', 'Vimentin.', 'SMA.', 'CD31.', 'CD56.', 'CD209.', 'CD11c.', 'CD11b.']


for d in (NPZ_DIR, MODEL_DIR, RESULTS_DIR):
    try:
        os.makedirs(os.path.join(d, PREFIX_SEG))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(os.path.join(d, PREFIX_CLASS))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(os.path.join(d, PREFIX_SAVE))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Create the training data for sample-based foreground/background segmentation
def generate_training_data_sample():
    raw_image_direc = 'raw'
    annotation_direc = 'annotated'

    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX_SEG),
        dimensionality=2,
        max_training_examples=MAX_TRAIN, # Define maximum number of training examples
        window_size_x=WINDOW_SIZE[0],
        window_size_y=WINDOW_SIZE[1],
        border_mode=BORDER_MODE,
        file_name_save=os.path.join(NPZ_DIR, PREFIX_SEG, FG_BG_DATA_FILE),
        training_direcs=TRAIN_DIR_SAMPLE,
        channel_names=CHANNELS_SEG,
        num_of_features=NUM_FEATURES_IN_SEG,
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc,
        reshape_size=RESHAPE_SIZE if RESIZE else None,
        edge_feature=[1, 0, 0], # Specify which feature is the edge feature,
        dilation_radius=1,
        output_mode=DATA_OUTPUT_MODE,
        display=False,
        verbose=True)

# Create training data for watershed energy transform
def generate_training_data_watershed():
    raw_image_direc = 'raw'
    annotation_direc = 'annotated'

    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX_SEG),
        dimensionality=2,
        max_training_examples=MAX_TRAIN, # Define maximum number of training examples
        window_size_x=WINDOW_SIZE[0],
        window_size_y=WINDOW_SIZE[1],
        border_mode=BORDER_MODE,
        file_name_save=os.path.join(NPZ_DIR, PREFIX_SEG, WATERSHED_DATA_FILE),
        training_direcs=TRAIN_DIR_SAMPLE,
        distance_transform=True,
        distance_bins=BINS,
        channel_names=CHANNELS_SEG,
        num_of_features=BINS,
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc,
        reshape_size=RESHAPE_SIZE if RESIZE else None,
        edge_feature=[1, 0, 0], # Specify which feature is the edge feature,
        dilation_radius=1,
        output_mode=DATA_OUTPUT_MODE,
        display=False,
        verbose=True)

# Create training data for classification
def generate_training_data_class():
    raw_image_direc = 'raw'
    annotation_direc = 'celltype'

    training_direcs = []
    for set_num in TRAIN_SET_RANGE:
        training_direcs.append('set' + str(set_num))
    if 'set30' in training_direcs: training_direcs.remove('set30')

   # Create the training data
    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX_CLASS),
        dimensionality=4,
        max_training_examples=MAX_TRAIN, # Define maximum number of training examples
        window_size_x=WINDOW_SIZE[0],
        window_size_y=WINDOW_SIZE[1],
        border_mode=BORDER_MODE,
        file_name_save=os.path.join(NPZ_DIR, PREFIX_CLASS, CLASS_DATA_FILE),
        training_direcs=training_direcs,
        channel_names=CHANNELS_CLASS,
        num_of_features=NUM_FEATURES_CLASS,
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc,
        reshape_size=RESHAPE_SIZE if RESIZE else None,
        edge_feature=[1], # Specify which feature is the edge feature, when possible put none to make balancing flexible
        dilation_radius=1,
        output_mode=DATA_OUTPUT_MODE,
        display=False,
        verbose=True)

# add in run_model_sample

# runs the sample and watershed segmentation models
def run_model_segmentation():
    raw_dir = 'raw'
    data_location = os.path.join(DATA_DIR, PREFIX_SEG, RUN_DIR, raw_dir)
    output_location = os.path.join(RESULTS_DIR, PREFIX_SAVE)
    image_size_x, image_size_y = get_image_sizes(data_location, CHANNELS_SEG)

    print('image_size_x is:', image_size_x)
    print('image_size_y is:', image_size_y)

    # define model type
    model_fn = dilated_bn_feature_net_31x31

    # Load the training data from NPZ into a numpy array
    testing_data = np.load(os.path.join(NPZ_DIR, PREFIX_SEG, CONV_DATA_FILE + '.npz'))

    X, y = testing_data['X'], testing_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    # save the size of the input data for input_shape model parameter
    size = X.shape[ROW_AXIS:COL_AXIS + 1]
    if IS_CHANNELS_FIRST:
        input_shape = (X.shape[CHANNEL_AXIS], size[0], size[1])
    else:
        input_shape = (size[0], size[1], len(CHANNELS_SEG))

    print(IS_CHANNELS_FIRST)
    print('input_shape is:', input_shape)

    watershed_weights_file = os.path.join(MODEL_DIR, PREFIX_SEG, MODEL_WSHED)
    fgbg_weights_file = os.path.join(MODEL_DIR, PREFIX_SEG, MODEL_FGBG)

    # load weights into both models
    run_watershed_model = model_fn(n_features=BINS, input_shape=input_shape)
    run_watershed_model.load_weights(watershed_weights_file)
    run_fgbg_model = model_fn(n_features=NUM_FEATURES_OUT_SEG, input_shape=input_shape)
    run_fgbg_model.load_weights(fgbg_weights_file)

    # get the data to run models on
    training_data_file = os.path.join(NPZ_DIR, PREFIX_SEG, CONV_DATA_FILE + '.npz')
    train_dict, (X_test, y_test) = get_data(training_data_file, mode='conv', seed=21)

    # run models
    test_images = run_watershed_model.predict(X_test)
    test_images_fgbg = run_fgbg_model.predict(X_test)

    print('watershed transform shape:', test_images.shape)
    print('segmentation mask shape:', test_images_fgbg.shape)

    argmax_images = []
    for i in range(test_images.shape[0]):
        argmax_images.append(np.argmax(test_images[i], axis=CHANNEL_AXIS))
    argmax_images = np.array(argmax_images)
    argmax_images = np.expand_dims(argmax_images, axis=CHANNEL_AXIS)

    print('watershed argmax shape:', argmax_images.shape)

    # threshold the foreground/background
    # and remove back ground from watershed transform
    if IS_CHANNELS_FIRST:
        fg_thresh = test_images_fgbg[:, 1, :, :] > 0.3
    else:
        fg_thresh = test_images_fgbg[:, :, :, 1] > 0.3

    fg_thresh = np.expand_dims(fg_thresh, axis=CHANNEL_AXIS)
    argmax_images_post_fgbg = argmax_images * fg_thresh

    print('reached')

    # Apply watershed method with the distance transform as seed
    from scipy import ndimage
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max

    watershed_images = []
    for i in range(argmax_images_post_fgbg.shape[0]):
        if IS_CHANNELS_FIRST:
            image = fg_thresh[i, 0, :, :]
            distance = argmax_images_post_fgbg[i, 0, :, :]
        else:
            image = fg_thresh[i, :, :, 0]
            distance = argmax_images_post_fgbg[i, :, :, 0]

        local_maxi = peak_local_max(distance, min_distance=10, indices=False, labels=image)

        markers = ndimage.label(local_maxi)[0]
        segments = watershed(-distance, markers, mask=image)
        watershed_images.append(segments)

    watershed_images = np.array(watershed_images)
    watershed_images = np.expand_dims(watershed_images, axis=CHANNEL_AXIS)

    index = 0

    tiff.imsave(os.path.join(output_location, 'raw_dsDNA.tif'), X_test[index, :, :, 0])
    tiff.imsave(os.path.join(output_location, 'seg_prediction.tif'), test_images_fgbg[index, :, :, 1])
    tiff.imsave(os.path.join(output_location, 'watershed_segmentation.tif'), watershed_images[index, :, :, 0])

    return watershed_images[index, :, :, 0]

# runs the classification model
def run_model_classification():
    raw_dir = 'raw'
    data_location = os.path.join(DATA_DIR, PREFIX_CLASS, RUN_DIR, raw_dir)
    # test_images = os.path.join(DATA_DIR, 'tissues/mibi/mibi_full/TNBCShareData', 'set1', raw_dir)
    output_location = os.path.join(RESULTS_DIR, PREFIX_CLASS)
    image_size_x, image_size_y = get_image_sizes(data_location, CHANNELS_CLASS)

    print('image_size_x is:', image_size_x)
    print('image_size_y is:', image_size_y)

    weights = os.path.join(MODEL_DIR, PREFIX_CLASS, MODEL_CLASS)

    if DATA_OUTPUT_MODE == 'sample':
        model_fn = dilated_bn_feature_net_31x31					#changed to 21x21
    elif DATA_OUTPUT_MODE == 'conv':
        model_fn = bn_dense_feature_net
    else:
        raise ValueError('{} is not a valid training mode for 2D images (yet).'.format(
            DATA_OUTPUT_MODE))

    predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=CHANNELS_CLASS,
        output_location=output_location,
        n_features=NUM_FEATURES_CLASS,
        model_fn=model_fn,
        list_of_weights=[weights],
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        win_x=WINDOW_SIZE[0],
        win_y=WINDOW_SIZE[1],
        split=False)

os.path.join(RESULTS_DIR, PREFIX_SAVE)

    for i in range(predictions.shape[0]):
        max_img = np.argmax(predictions[i], axis=-1)
        max_img = max_img.astype(np.int16)
        cnnout_name = 'pipe_class_argmax_frame_{}.tif'.format(str(i).zfill(3))
        out_file_path = os.path.join(output_location, cnnout_name)
        tiff.imsave(out_file_path, max_img)

    return max_img

def post_processing(instance, classification):

    # make an empty array of the same size as the instance input to store the output values
    rows = instance.shape[0]
    cols = instance.shape[1]
    output = np.zeros((cols, rows), dtype='uint16')

    # for each unique cell
    for label in range(1, (instance.max()+1)):

        label_classes = np.array([])
        pixel_locations = np.argwhere(instance == label)

        for x_y in pixel_locations:

            x = x_y[1]
            y = x_y[0]
            label_classes = np.append(label_classes, classification[y,x])

        # find the most prevalent class in the cell
        m = stats.mode(label_classes)
        cell_class = np.asscalar(m[0])
        print('class of cell#', label, ' is:', cell_class)

        for x_y in pixel_locations:

            x = x_y[1]
            y = x_y[0]

            output[y,x] = cell_class

    tiff.imsave('./output.tif', output)


# runs model on segmentation/watershed/classification, and postprocesses the results.
def run_pipeline_on_dir():
    instance_seg = run_model_segmentation()
    cell_classes = run_model_classification()

    post_processing(instance_seg, cell_classes)
