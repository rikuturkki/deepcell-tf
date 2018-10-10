import numpy as np
import skimage.io as sk
import tifffile as tiff
import os

SET_RANGE = range(1, 44+1)

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'

PREFIX_SEG = 'tissues/mibi/samir'
PREFIX_CLASS = 'tissues/mibi/mibi_full'
PREFIX_SAVE = 'tissues/mibi/pipeline'


NUM_FEATURES_IN_SEG = 2
NUM_FEATURES_OUT_SEG = 3
NUM_FEATURES_CLASS = 17

MODEL_FGBG = '2018-07-13_mibi_31x31_channels_last_sample__0.h5'
CHANNELS_SEG = ['dsDNA', 'Ca', 'H3K27me3', 'H3K9ac', 'Ta']

DTYPE = 'float32'
WIN_SIZE = (15, 15)
NUM_CHAN = 3



def concat_channels(dsDNA, f0, f1):

    # make inputs compatible
    dsDNA = dsDNA.astype(DTYPE)
    #dsDNA = dsDNA / dsDNA.max()
    f0 = f0.astype(DTYPE)
    f1 = f1.astype(DTYPE)

    # print image info
    print('dsDNA shape is:', dsDNA.shape, ', type is:', dsDNA.dtype, ' max is:', dsDNA.max())
    print('feature shape is:', f0.shape, ', type is:', f0.dtype, ' max is:', f0.max())

    # pad images back to 2048 x 2048
    dsDNA = np.pad(dsDNA, WIN_SIZE, 'constant')
    f0 = np.pad(f0, WIN_SIZE, 'constant')
    f1 = np.pad(f1, WIN_SIZE, 'constant')

    # make empty output image
    output = np.zeros((dsDNA.shape[0], dsDNA.shape[1], NUM_CHAN), dtype=DTYPE)

    print('padded feature shape is:', f0.shape)
    print('output shape is:', output.shape)
    print('')

    # insert each layer into output
    output[:,:,0] = dsDNA[:,:]
    output[:,:,1] = f0[:,:]
    output[:,:,2] = f1[:,:]

    return output

def run_segmentation(set):
    raw_dir = 'raw'
    data_location = os.path.join(DATA_DIR, PREFIX_CLASS, set, raw_dir)
    output_location = os.path.join(RESULTS_DIR, PREFIX_SEG)
    image_size_x, image_size_y = get_image_sizes(data_location, channel_names)

    weights = os.path.join(MODEL_DIR, PREFIX_SEG, MODEL_FGBG)

    n_features = 3
    window_size = (30, 30)

    if DATA_OUTPUT_MODE == 'sample':
        model_fn = dilated_bn_feature_net_31x31                                 #changed to 21x21
    elif DATA_OUTPUT_MODE == 'conv':
        model_fn = bn_dense_feature_net
    else:
        raise ValueError('{} is not a valid training mode for 2D images (yet).'.format(
            DATA_OUTPUT_MODE))

    predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=CHANNELS_SEG,
        output_location=output_location,
        n_features=n_features,
        model_fn=model_fn,
        list_of_weights=[weights],
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        win_x=WINDOW_SIZE[0],
        win_y=WINDOW_SIZE[1],
        split=False)

    #0.25 0.25 works good
    edge_thresh = EDGE_THRESH
    interior_thresh = INT_THRESH
    cell_thresh = CELL_THRESH

    print('shape of predictions is:', predictions.shape)

    edge = np.copy(predictions[:,:,:,0])
    edge[edge < edge_thresh] = 0
    edge[edge > edge_thresh] = 1

    interior = np.copy(predictions[:, :, :, 1])
    interior[interior > interior_thresh] = 1
    interior[interior < interior_thresh] = 0

    cell_notcell = 1 - np.copy(predictions[:, :, :, 2])
    cell_notcell[cell_notcell > cell_thresh] = 1
    cell_notcell[cell_notcell < cell_thresh] = 0

    # define foreground as the interior bounded by edge
    fg_thresh = np.logical_and(interior==1, edge==0)

    # remove small objects from the foreground segmentation
    fg_thresh = skimage.morphology.remove_small_objects(fg_thresh, min_size=50, connectivity=1)

    #fg_thresh = skimage.morphology.binary_erosion(fg_thresh)
    #fg_thresh = skimage.morphology.binary_dilation(fg_thresh)

    fg_thresh = np.expand_dims(fg_thresh, axis=CHANNEL_AXIS)

    watershed_segmentation = skimage.measure.label(  np.squeeze(fg_thresh), connectivity=2)


    # dilate gradually into the mask area
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)

    watershed_segmentation = dilate_nomask(watershed_segmentation, 1)
    watershed_segmentation = erode(watershed_segmentation, 2)
    watershed_segmentation = dilate_nomask(watershed_segmentation, 2)
    watershed_segmentation = erode(watershed_segmentation, NUM_FINAL_EROSIONS)

    index = 0

    output_location = os.path.join(RESULTS_DIR, PREFIX_SAVE)
    print('saving to: ', output_location)

    dsDNA = tiff.imread(os.path.join(data_location, 'dsDNA.tif'))
    dsDNA = dsDNA[15:-15, 15:-15]

    watershed_segmentation = watershed_segmentation.astype('float32')

    cell_edge = predictions[index, :, :, 0]

    return watershed_segmentation, dsDNA, cell_notcell, cell_edge

def crop_im_and_save(img, save_file_path, im_size, crop_size, im_name, mask):

    crop_counter = 0
    for x in range(0, im_size, crop_size):
        for y in range(0, im_size, crop_size):

            crop = np.zeros((crop_size, crop_size))
            crop = img[x:x+crop_size, y:y+crop_size]
            crop_num = str(crop_counter)
            out_file_path = save_file_path + '/' + mask + 'crop' + crop_num + '_' + im_name
            tiff.imsave( out_file_path, crop)
            crop_counter += 1

def process_set(set_num):

    set_dir = 'set' + str(set_num)
    mask, raw, interior, edge = run_segmentation(set_dir)

    # concatenate raw, interior, and edge into one 3 channel image
    output = concat_channels(raw, interior, edge)

    output_name = 'dsDNA' + str(set_num) + '.tif'
    mask_name = 'mask_dsDNA' + str(set_num) + '.tif'

    crop_im_and_save(output, os.path.join(DATA_DIR, TRAIN_DIR), 2048, 256, output_name, mask='')
    crop_im_and_save(mask, os.path.join(DATA_DIR, ANNO_DIR), 2048, 256, output_name, mask='mask_')

if __name__ == '__main__':

    # make storage directories
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
        os.mkdir(os.path.join(DATA_DIR, ANNO_DIR))
        os.mkdir(os.path.join(DATA_DIR, TRAIN_DIR))
        os.mkdir(os.path.join(DATA_DIR, TEST_DIR))

    # generate training data for all sets
    for set_num in SET_RANGE:
        if set_num == 2:
            continue
        else:
            process(set_num)
