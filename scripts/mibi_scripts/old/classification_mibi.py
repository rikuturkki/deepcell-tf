#classification sample based

## Generate training data
import os                   #operating system interface
import errno                #error symbols
import argparse             #command line input parsing

import numpy as np          #scientific computing (aka matlab)
import skimage.external.tifffile as tiff     #read/write TIFF files (aka our images)
from tensorflow.python.keras.optimizers import SGD    #optimizer
from tensorflow.python.keras import backend as K            #tensorflow backend

from deepcell import get_image_sizes                #io_utils, returns shape of first image inside data_location
from deepcell import make_training_data             #data_utils, reads images in training directories and saves as npz file
from deepcell import bn_feature_net_31x31           #model_zoo
from deepcell import dilated_bn_feature_net_31x31

from deepcell import bn_dense_feature_net
from deepcell import rate_scheduler                 #train_utils,
from deepcell import train_model_disc, train_model_conv, train_model_sample     #training.py, probably use sample
from deepcell import run_models_on_directory
from deepcell import export_model

# data options
#DATA_OUTPUT_MODE = 'conv'
DATA_OUTPUT_MODE = 'sample'
BORDER_MODE = 'valid' if DATA_OUTPUT_MODE == 'sample' else 'same'
RESIZE = True
RESHAPE_SIZE = 2048 
N_EPOCHS = 5
WINDOW_SIZE = (15,15)
BATCH_SIZE = 32 

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'
#PREFIX = 'tissues/mibi/samir'
PREFIX = 'tissues/mibi/mibi_full'
DATA_FILE = 'mibi_medov4_class_{}_{}'.format(K.image_data_format(), DATA_OUTPUT_MODE)
MODEL_NAME = '2018-08-01_mibi_medov4_class_channels_last_sample__0.h5'

RUN_DIR = 'set1'

TRAIN_SET_RANGE = range(1, 1+1) 


MAX_TRAIN = 4e5
#CHANNEL_NAMES = ['dsDNA', 'Ca', 'H3K27me3', 'H3K9ac', 'Ta', 'P']  #Add P?
#CHANNEL_NAMES = ['dsDNA']

# removed channels: CSF-1R, CD-163, B7H3
'''CHANNEL_NAMES = ['Au', 'catenin', 'Ca', 'CD11b', 'CD11c', 'CD138', 'CD16', 'CD209', 'CD20', 'CD31', 
                 'CD3', 'CD45RO', 'CD45', 'CD4', 'CD56', 'CD63.', 'CD68.', 'CD8.', 'C.', 'dsDNA.', 
                 'EGFR.', 'Fe.', 'FoxP3.', 'H3K27me3.', 'H3K9ac.', 'HLA_Class_1.', 'HLA-DR.', 'IDO.', 
                 'Keratin17.', 'Keratin6.', 'Ki67.', 'Lag3.', 'MPO.', 'Na.', 'OX40.', 'p53.', 'Pan-Keratin.', 
                 'PD1.', 'PD-L1.', 'phospho-S6.', 'P.', 'Si.', 'SMA.', 'Ta.', 'Vimentin.', ]'''

CHANNEL_NAMES = ['dsDNA.', 'H3K27me3.', 'Ta.', 'FoxP3.', 'CD4.', 'CD16.', 'EGFR.', 'CD68.', 'CD8.', 'CD3.', 
                 'Keratin17.', 'CD20.', 'p53.', 'catenin.', 'HLA-DR.', 'CD45.', 'Pan-Keratin.', 'MPO.', 
                 'Keratin6.', 'Vimentin.', 'SMA.', 'CD31.', 'CD56.', 'CD209.', 'CD11c.', 'CD11b.']
NUM_FEATURES = 17


for d in (NPZ_DIR, MODEL_DIR, RESULTS_DIR):
    try:
        os.makedirs(os.path.join(d, PREFIX))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

def generate_training_data():
    file_name_save = os.path.join(NPZ_DIR, PREFIX, DATA_FILE)
    num_of_features = 17 # Specify the number of feature masks that are present   
    channel_names = CHANNEL_NAMES
    raw_image_direc = 'raw'
    annotation_direc = 'celltype'

    training_direcs = []
    for set_num in TRAIN_SET_RANGE:
        training_direcs.append('set' + str(set_num))

   # Create the training data
    make_training_data(
        direc_name=os.path.join(DATA_DIR, PREFIX),
        dimensionality=4,
        max_training_examples=MAX_TRAIN, # Define maximum number of training examples
        window_size_x=WINDOW_SIZE[0],
        window_size_y=WINDOW_SIZE[1],
        border_mode=BORDER_MODE,
        file_name_save=file_name_save,
        training_direcs=training_direcs,
        channel_names=CHANNEL_NAMES,
        num_of_features=num_of_features,
        raw_image_direc=raw_image_direc,
        annotation_direc=annotation_direc,
        reshape_size=RESHAPE_SIZE if RESIZE else None,
        edge_feature=[1], # Specify which feature is the edge feature,
        dilation_radius=1,
        output_mode=DATA_OUTPUT_MODE,
        display=False,
        verbose=True)

def train_model_on_training_data():
    direc_save = os.path.join(MODEL_DIR, PREFIX)
    direc_data = os.path.join(NPZ_DIR, PREFIX)
    training_data = np.load(os.path.join(direc_data, DATA_FILE + '.npz'))

    class_weights = training_data['class_weights']
    X, y = training_data['X'], training_data['y']
    print('X.shape: {}\ny.shape: {}'.format(X.shape, y.shape))

    n_epoch = N_EPOCHS
    batch_size = BATCH_SIZE if DATA_OUTPUT_MODE == 'sample' else 1
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    lr_sched = rate_scheduler(lr=0.01, decay=0.99)

    model_args = {
        'norm_method': None,
        'reg': 1e-5,
        'n_features': 17,
    }

    data_format = K.image_data_format()
    row_axis = 2 if data_format == 'channels_first' else 1
    col_axis = 3 if data_format == 'channels_first' else 2
    channel_axis = 1 if data_format == 'channels_first' else 3

    if DATA_OUTPUT_MODE == 'sample':
        train_model = train_model_sample
        the_model = bn_feature_net_31x31				#changed to 21x21
        model_args['n_channels'] = len(CHANNEL_NAMES)

    elif DATA_OUTPUT_MODE == 'conv' or DATA_OUTPUT_MODE == 'disc':
        train_model = train_model_conv
        the_model = bn_dense_feature_net
        model_args['location'] = False

        size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
        if data_format == 'channels_first':
            model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
        else:
            model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])

    model = the_model(**model_args)

    train_model(
        model=model,
        dataset=DATA_FILE,
        optimizer=optimizer,
        batch_size=batch_size,
        n_epoch=n_epoch,
        direc_save=direc_save,
        direc_data=direc_data,
        lr_sched=lr_sched,
        class_weight=class_weights,
        rotation_range=180,
        flip=True,
        shear=True)


def run_model_on_dir():
    raw_dir = 'raw'
    data_location = os.path.join(DATA_DIR, PREFIX, 'set1', raw_dir)
#    test_images = os.path.join(DATA_DIR, 'tissues/mibi/mibi_full/TNBCShareData', 'set1', raw_dir)
    output_location = os.path.join(RESULTS_DIR, PREFIX)
    image_size_x, image_size_y = get_image_sizes(data_location, CHANNEL_NAMES)


    print('image_size_x is:', image_size_x)
    print('image_size_y is:', image_size_y)

#    model_name = '2018-07-13_mibi_31x31_{}_{}__0.h5'.format(
#        K.image_data_format(), DATA_OUTPUT_MODE)

    weights = os.path.join(MODEL_DIR, PREFIX, MODEL_NAME)

    n_features = 17  

    if DATA_OUTPUT_MODE == 'sample':
        model_fn = dilated_bn_feature_net_31x31					#changed to 21x21
    elif DATA_OUTPUT_MODE == 'conv':
        model_fn = bn_dense_feature_net
    else:
        raise ValueError('{} is not a valid training mode for 2D images (yet).'.format(
            DATA_OUTPUT_MODE))

    predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=CHANNEL_NAMES,
        output_location=output_location,
        n_features=n_features,
        model_fn=model_fn,
        list_of_weights=[weights],
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        win_x=WINDOW_SIZE[0],
        win_y=WINDOW_SIZE[1],
        split=False)

    for i in range(predictions.shape[0]):
        max_img = np.argmax(predictions[i], axis=-1)
        max_img = max_img.astype(np.int16)
        cnnout_name = 'argmax_frame_{}.tif'.format(str(i).zfill(3))

        out_file_path = os.path.join(output_location, cnnout_name)

        tiff.imsave(out_file_path, max_img)


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
        the_model = dilated_bn_feature_net_31x31
        if K.image_data_format() == 'channels_first':
            model_args['input_shape'] = (1, 1080, 1280)
        else:
            model_args['input_shape'] = (1080, 1280, 1)

    elif DATA_OUTPUT_MODE == 'conv' or DATA_OUTPUT_MODE == 'disc':
        the_model = bn_dense_feature_net
        model_args['location'] = False

        size = (RESHAPE_SIZE, RESHAPE_SIZE) if RESIZE else X.shape[row_axis:col_axis + 1]
        if data_format == 'channels_first':
            model_args['input_shape'] = (X.shape[channel_axis], size[0], size[1])
        else:
            model_args['input_shape'] = (size[0], size[1], X.shape[channel_axis])

    model = the_model(**model_args)

    model_name = '2018-06-27_mibi_samir_{}_{}__0.h5'.format(
        K.image_data_format(), DATA_OUTPUT_MODE)

    weights_path = os.path.join(MODEL_DIR, PREFIX, model_name)
    export_path = os.path.join(EXPORT_DIR, PREFIX)
    export_model(model, export_path, model_version=0, weights_path=weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=['train', 'run', 'export'],
                        help='train or run models')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        help='force re-write of training data npz files')

    args = parser.parse_args()

    if args.command == 'train':
        data_file_exists = os.path.isfile(os.path.join(NPZ_DIR, PREFIX, DATA_FILE + '.npz'))
        if args.overwrite or not data_file_exists:
            generate_training_data()

        train_model_on_training_data()

    elif args.command == 'run':
        run_model_on_dir()

    elif args.command == 'export':
        export()
