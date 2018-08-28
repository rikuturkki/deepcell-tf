#Sample watershed

## Generate training data
import os
import errno
import argparse

import numpy as np
import skimage.external.tifffile as tiff
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import backend as K

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
#DATA_OUTPUT_MODE = 'conv'
DATA_OUTPUT_MODE = 'sample'
BORDER_MODE = 'valid' if DATA_OUTPUT_MODE == 'sample' else 'same'
RESIZE = True                                                              #was True
RESHAPE_SIZE = 512
WINDOW_SIZE = (15,15)
N_EPOCHS = 20
BINS = 4
MAX_TRAIN = 1e7
CHANNEL_NAMES = ['dsDNA', 'Ca', 'H3K27me3', 'H3K9ac', 'Ta']

# Check for channels_first or channels_last
IS_CHANNELS_FIRST = K.image_data_format() == 'channels_first'
ROW_AXIS = 2 if IS_CHANNELS_FIRST else 1
COL_AXIS = 3 if IS_CHANNELS_FIRST else 2
CHANNEL_AXIS = 1 if IS_CHANNELS_FIRST else -1

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'
PREFIX = 'tissues/mibi/samir'
FG_BG_DATA_FILE = 'mibi_watershedFB_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)
WATERSHED_DATA_FILE = 'mibi_watershed_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)
CONV_DATA_FILE = 'mibi_watershedconv_{}_{}'.format(K.image_data_format(), 'conv')
RUN_DIR = 'set1'

#MODEL_FGBG_OLD '2018-07-20_mibi_watershedFB_channels_last_sample_fgbg_0.h5'
#MODEL_WSHED_OLD = '2018-07-20_mibi_watershed_channels_last_sample_watershed_0.h5'
#CHANNEL_NAMES_OLD = ['dsDNA', 'Ca', 'H3K27me3', 'H3K9ac', 'Ta', 'P.']

MODEL_FGBG = '2018-08-02_mibi_watershedFB_channels_last_sample_fgbg_0.h5'
MODEL_WSHED = '2018-08-03_mibi_watershed_channels_last_sample_watershed_0.h5'

for d in (NPZ_DIR, MODEL_DIR, RESULTS_DIR):
    try:
        os.makedirs(os.path.join(d, PREFIX))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

def generate_training_data():
#    file_name_save = os.path.join(NPZ_DIR, PREFIX, DATA_FILE)
    num_of_features = 1 # Specify the number of feature masks that are present
    training_direcs = ['set1', 'set2']
    channel_names = CHANNEL_NAMES
    raw_image_direc = 'raw'
    annotation_direc = 'annotated'

    # Create the training data for foreground/background segmentation
    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX),
        dimensionality=2,
        max_training_examples=MAX_TRAIN, # Define maximum number of training examples
        window_size_x=WINDOW_SIZE[0],
        window_size_y=WINDOW_SIZE[1],
        border_mode=BORDER_MODE,
        file_name_save=os.path.join(NPZ_DIR, PREFIX, FG_BG_DATA_FILE),
        training_direcs=training_direcs,
        distance_transform=False,
        distance_bins=BINS,
        channel_names=channel_names,
        num_of_features=BINS,
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc,
        reshape_size=RESHAPE_SIZE if RESIZE else None,
        edge_feature=[1, 0, 0], # Specify which feature is the edge feature,
        dilation_radius=1,
        output_mode=DATA_OUTPUT_MODE,
        display=False,
        verbose=True)

    # Create training data for watershed energy transform
    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX),
        dimensionality=2,
        max_training_examples=MAX_TRAIN, # Define maximum number of training examples
        window_size_x=WINDOW_SIZE[0],
        window_size_y=WINDOW_SIZE[1],
        border_mode=BORDER_MODE,
        file_name_save=os.path.join(NPZ_DIR, PREFIX, WATERSHED_DATA_FILE),
        training_direcs=training_direcs,
        distance_transform=True,
        distance_bins=BINS,
        channel_names=channel_names,
        num_of_features=BINS,
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc,
        reshape_size=RESHAPE_SIZE if RESIZE else None,
        edge_feature=[1, 0, 0], # Specify which feature is the edge feature,
        dilation_radius=1,
        output_mode=DATA_OUTPUT_MODE,
        display=False,
        verbose=True)

def train_model_on_training_data():
    direc_save = os.path.join(MODEL_DIR, PREFIX)
    direc_data = os.path.join(NPZ_DIR, PREFIX)
    training_data = np.load(os.path.join(direc_data,FG_BG_DATA_FILE + '.npz'))

    #class_weights = training_data['class_weights']
    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    batch_size = 32 if DATA_OUTPUT_MODE == 'sample' else 1
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    lr_sched = rate_scheduler(lr=0.01, decay=0.99)
    n_epoch=N_EPOCHS

    distance_bins = 4

    model_args = {
        'norm_method': 'max',
        'reg': 1e-5,
        'n_features': distance_bins,
        'n_channels' : len(CHANNEL_NAMES)
    }

    data_format = K.image_data_format()
    row_axis = 2 if data_format == 'channels_first' else 1
    col_axis = 3 if data_format == 'channels_first' else 2
    channel_axis = 1 if data_format == 'channels_first' else 3

    size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]  #added

    if data_format == 'channels_first':
        model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
    else:
        model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])

    #instantiate and train foreground/background separation model
    fgbg_model = bn_feature_net_31x31(n_features=3, n_channels=len(CHANNEL_NAMES))

    train_model_sample(
        model=fgbg_model,
        dataset=FG_BG_DATA_FILE,
        optimizer=optimizer,
        batch_size=batch_size,
        n_epoch=N_EPOCHS,
        direc_save=os.path.join(MODEL_DIR, PREFIX),
        direc_data=os.path.join(NPZ_DIR, PREFIX),
        expt='fgbg',
        lr_sched=lr_sched,
        class_weight=training_data['class_weights'],
        rotation_range=180,
        flip=True,
        shear=False)


    # instantiate and train watershed model
    watershed_model = bn_feature_net_31x31(n_features=BINS, n_channels=len(CHANNEL_NAMES))

    train_model_watershed_sample(
        model=watershed_model,
        dataset=WATERSHED_DATA_FILE,
        optimizer=optimizer,
        batch_size=batch_size,
        n_epoch=n_epoch,
        distance_bins=BINS,
        direc_save=os.path.join(MODEL_DIR, PREFIX),
        direc_data=os.path.join(NPZ_DIR, PREFIX),
        expt='watershed',
        lr_sched=lr_sched,
        class_weight=training_data['class_weights'],
        rotation_range=180,
        flip=True,
        shear=False)

def run_model_on_dir(generate_conv):
    raw_dir = 'raw'
    data_location = os.path.join(DATA_DIR, PREFIX_SEG, RUN_DIR, raw_dir)
    output_location = os.path.join(RESULTS_DIR, PREFIX_SEG)
    channel_names = CHANNELS_SEG
    image_size_x, image_size_y = get_image_sizes(data_location, channel_names)



    # Make conv training data for mask generation
    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX_SEG),
        dimensionality=2,
        max_training_examples=MAX_TRAIN,
        window_size_x=WINDOW_SIZE[0],
        window_size_y=WINDOW_SIZE[1],
        border_mode=BORDER_MODE,
        file_name_save=os.path.join(NPZ_DIR, PREFIX_SEG, CONV_DATA_FILE),
        training_direcs=['set2'],
        distance_transform=False,  # not needed for conv mode
        distance_bins=BINS,  # not needed for conv mode
        channel_names=CHANNEL_NAMES,
        num_of_features=BINS,
        raw_image_direc='raw',
        annotation_direc='annotated',
        reshape_size=None,
        edge_feature=[1, 0, 0],
        dilation_radius=1,
        output_mode='conv',
        display=False,
        verbose=True)



    print('image_size_x is:', image_size_x)
    print('image_size_y is:', image_size_y)


    # define model type
    model_fn = dilated_bn_feature_net_31x31

    # model names
    watershed_weights_file = MODEL_WSHED
    watershed_weights_file = os.path.join(MODEL_DIR, PREFIX, watershed_weights_file)

    # weights directories
   # fgbg_weights_file = '2018-07-13_mibi_watershedFB_channels_last_sample_fgbg_0.h5'
    fgbg_weights_file = MODEL_FGBG
    fgbg_weights_file = os.path.join(MODEL_DIR, PREFIX_SEG, fgbg_weights_file)

    # variables
    n_features = 4
    window_size = WINDOW_SIZE

    # Load the training data from NPZ into a numpy array
    testing_data = np.load(os.path.join(NPZ_DIR, PREFIX_SEG, CONV_DATA_FILE + '.npz'))

    X, y = testing_data['X'], testing_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    # save the size of the input data for input_shape model parameter
    #size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[ROW_AXIS:COL_AXIS + 1]
    size = X.shape[ROW_AXIS:COL_AXIS + 1]
    if IS_CHANNELS_FIRST:
        input_shape = (X.shape[CHANNEL_AXIS], size[0], size[1])
    else:
   #     input_shape = (size[0], size[1], X.shape[CHANNEL_AXIS])
        input_shape = (size[0], size[1], len(CHANNEL_NAMES))

    print(IS_CHANNELS_FIRST)
    print('input_shape is:', input_shape)

    # load weights into both models
    run_watershed_model = model_fn(n_features=BINS, input_shape=input_shape)
    run_watershed_model.load_weights(watershed_weights_file)
    run_fgbg_model = model_fn(n_features=3, input_shape=input_shape)
    run_fgbg_model.load_weights(fgbg_weights_file)

    # get the data to run models on
    training_data_file = os.path.join(NPZ_DIR, PREFIX, CONV_DATA_FILE + '.npz')
    train_dict, (X_test, y_test) = get_data(training_data_file, mode='conv', seed=21)

    # run models
    test_images = run_watershed_model.predict(X_test)
    test_images_fgbg = run_fgbg_model.predict(X_test)



    print('watershed transform shape:', test_images.shape)
    print('segmentation mask shape:', test_images_fgbg.shape)

    print('max of test_edge is: ', test_images_fgbg[:, :, :, 0])

    argmax_images = []
    for i in range(test_images.shape[0]):
        argmax_images.append(np.argmax(test_images[i], axis=CHANNEL_AXIS))
    argmax_images = np.array(argmax_images)
    argmax_images = np.expand_dims(argmax_images, axis=CHANNEL_AXIS)

    print('watershed argmax shape:', argmax_images.shape)

    # threshold the foreground/background
    # and remove back ground from watershed transform
    #fg_thresh = (test_images_fgbg[:, :, :, 0] + test_images_fgbg[:, :, :, 1] ) > 0.4
    fg_thresh = test_images_fgbg[:, :, :, 1] > 0.3


#    if IS_CHANNELS_FIRST:
#        fg_thresh = test_images_fgbg[:, 1, :, :] > 0.8
#    else:
#        fg_thresh = test_images_fgbg[:, :, :, 1] > 0.35

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

    tiff.imsave(os.path.join(output_location, 'source.tif'), X_test[index, :, :, 0])
    tiff.imsave(os.path.join(output_location, 'int_prediction.tif'), test_images_fgbg[:, :, :, 1])
    tiff.imsave(os.path.join(output_location, 'edge_prediction.tif'), test_images_fgbg[:, :, :, 0])
#    tiff.imsave(os.path.join(output_location, 'thresholded_seg.tif'), fg_thresh[index, iii:, ])]
#    tiff.imsave(os.path.join(output_location, 'watershed_transform.tif'), argmax_images[index, :, :, 0])
#    tiff.imsave(os.path.join(output_location, 'watershed_transform_no_backg.tif'), argmax_images_post_fgbg[index, :, :, 0])
    tiff.imsave(os.path.join(output_location, 'Watershed_Segmentation.tif'), watershed_images[index, :, :, 0])


'''
    predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=channel_names,
        output_location=output_location,
        n_features=n_features,
        model_fn=model_fn,
        list_of_weights=[weights],
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        win_x=window_size[0],
        win_y=window_size[1],
        split=True)



#    import pdb; pdb.set_trace()

    for i in range(predictions.shape[0]):
        max_img = np.argmax(predictions[i], axis=-1)
        max_img = max_img.astype(np.int16)
        cnnout_name = 'argmax_frame_{}.tif'.format(str(i).zfill(3))

        out_file_path = os.path.join(output_location, cnnout_name)

        tiff.imsave(out_file_path, max_img)
'''


'''
def export():
    model_args = {
        'norm_method': 'median',
        'reg': 1e-5,
        'n_features': 3
    }

    direc_data = os.path.join(NPZ_DIR, PREFIX)
    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))
    X, y = training_data['X'], training_data['y']

    data_format = K.image_data_format()
    row_axis = 2 if data_format == 'channels_first' else 1
    col_axis = 3 if data_format == 'channels_first' else 2
    channel_axis = 1 if data_format == 'channels_first' else 3

    if DATA_OUTPUT_MODE == 'sample':
        the_model = watershednetwork
        if K.image_data_format() == 'channels_first':
            model_args['input_shape'] = (1, 1080, 1280)
        else:
            model_args['input_shape'] = (1080, 1280, 1)

    elif DATA_OUTPUT_MODE == 'conv' or DATA_OUTPUT_MODE == 'disc':
        the_model = watershednetwork
        model_args['location'] = False

        size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
        if data_format == 'channels_first':
            model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
        else:
            model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])

    model = the_model(**model_args)

    model_name = '2018-07-06_mibi_watershed_{}_{}__0.h5'.format(
        K.image_data_format(), DATA_OUTPUT_MODE)

    weights_path = os.path.join(MODEL_DIR, PREFIX, model_name)
    export_path = os.path.join(EXPORT_DIR, PREFIX)
    export_model(model, export_path, model_version=0, weights_path=weights_path)
'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=['train', 'run', 'export'],
                        help='train or run models')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        help='force re-write of training data npz files')

    args = parser.parse_args()

    if args.command == 'train':
        data_file_exists = os.path.isfile(os.path.join(NPZ_DIR, PREFIX, FG_BG_DATA_FILE + '.npz'))
        if args.overwrite or not data_file_exists:
            generate_training_data()

        train_model_on_training_data()

    elif args.command == 'run':

        generate_conv = False

        if args.overwrite:
            generate_conv = True

        run_model_on_dir(generate_conv)
