import numpy as np
import skimage.io as sk
import tifffile as tiff
import os



DSDNA = 'channel_dsDNA.tif'
EDGE = 'feature_0_frame_0.tif'
INTERIOR = 'feature_1_frame_0.tif'
DTYPE = 'float32'
WIN_SIZE = (15, 15)
NUM_CHAN = 3
MASK = 'interior.tif'


DATA_DIR = './data'
ANNO_DIR = 'annotated'
TRAIN_DIR = 'raw_train'
TEST_DIR = 'raw_test'

def concat_channels(SET_DIR): 
    # read images 
    dsDNA_raw = sk.imread(os.path.join(SET_DIR, DSDNA))
    f0 = sk.imread(os.path.join(SET_DIR, EDGE))
    f1 = sk.imread(os.path.join(SET_DIR, INTERIOR))

    # print image info
    print('dsDNA shape is:', dsDNA_raw.shape, ', type is:', dsDNA_raw.dtype, ' max is:', dsDNA_raw.max())
    print('feature shape is:', f0.shape, ', type is:', f0.dtype, ' max is:', f0.max())
    print('')

    # make inputs compatible
    dsDNA = dsDNA_raw[15:-15, 15:-15]
    dsDNA = dsDNA.astype(DTYPE)
    dsDNA = dsDNA / dsDNA.max()
    f0 = f0.astype(DTYPE)
    f1 = f1.astype(DTYPE)

    # print image info 
    print('dsDNA shape is:', dsDNA.shape, ', type is:', dsDNA.dtype, ' max is:', dsDNA.max())
    print('feature shape is:', f0.shape, ', type is:', f0.dtype, ' max is:', f0.max())
    print('')

    # pad images back to 2048 x 2048
    dsDNA = np.pad(dsDNA, (WIN_SIZE, WIN_SIZE),'constant')
    f0 = np.pad(f0, (WIN_SIZE, WIN_SIZE), 'constant')
    f1 = np.pad(f1, (WIN_SIZE, WIN_SIZE), 'constant')

    # make empty output image
    output = np.zeros((dsDNA.shape[0], dsDNA.shape[1], NUM_CHAN), dtype=DTYPE)

    print('padded feature shape is:', f0.shape)
    print('output shape is:', output.shape)

    # insert each layer into output
    output[:,:,0] = dsDNA[:,:]
    output[:,:,1] = f0[:,:]
    output[:,:,2] = f1[:,:]

    return output


def trim_mask(SET_DIR):
    mask = sk.imread(os.path.join(SET_DIR, MASK))
    mask = mask[15:-15, 15:-15]
    mask = np.pad(mask, (WIN_SIZE, WIN_SIZE), 'constant')
    return mask


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
 
    output = concat_channels(set_dir)
    mask = trim_mask(set_dir)

    output_name = 'dsDNA' + str(set_num) + '.tif'
    mask_name = 'mask_dsDNA' + str(set_num) + '.tif'

    crop_im_and_save(output, os.path.join(DATA_DIR, TRAIN_DIR), 2048, 512, output_name, mask='')
    crop_im_and_save(mask, os.path.join(DATA_DIR, ANNO_DIR), 2048, 512, output_name, mask='mask_')



if __name__ == '__main__':

    # make storage directories
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
        os.mkdir(os.path.join(DATA_DIR, ANNO_DIR))
        os.mkdir(os.path.join(DATA_DIR, TRAIN_DIR))
        os.mkdir(os.path.join(DATA_DIR, TEST_DIR))

    process_set(1)
    process_set(2)
 









