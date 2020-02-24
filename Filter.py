from __future__ import print_function

import copy
import cv2
from numba import cuda
import numpy as np
import os

@cuda.jit
def k_filter_horizontal(mask, img):
    ty = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y

    # Check if the current thread is out of the region.
    if ( ty >= mask.shape[0] ):
        return

    # Process the image line with row index ty in groups of 3 pixels.
    for x in range( 1, mask.shape[1] - 1 ):
        if ( mask[ty, x-1] == mask[ty, x] or mask[ty, x] == mask[ty, x+1]):
            # Median value is the current pixel. Do nothing
            pass
        else:
            # Median value is not the current pixel. 

            # Copy value the median value to the current mask location.
            mask[ty, x] = mask[ty, x-1]

            # Assign average value to the current image location.
            img[ty, x] = 0.5*img[ty, x-1] + 0.5*img[ty, x+1]

@cuda.jit
def k_filter_vertical(mask, img):
    tx = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x

    # Check if the current thread is out of the region.
    if ( tx >= mask.shape[1] ):
        return

    # Process the image line with row index ty in groups of 3 pixels.
    for y in range( 1, mask.shape[0] - 1 ):
        if ( mask[y-1, tx] == mask[y, tx] or mask[y+1, tx] == mask[y, tx]):
            # Median value is the current pixel. Do nothing
            pass
        else:
            # Median value is not the current pixel. 

            # Copy value the median value to the current mask location.
            mask[y, tx] = mask[y-1, tx]

            # Assign average value to the current image location.
            img[y, tx] = 0.5*img[y-1, tx] + 0.5*img[y+1, tx]

def guided_filter_3(mask, img):
    """
    mask: A 1-channel mask image, numpy array. dtype == np.uint8. Two-value mask.
    img: A 1-channel image, numpy array.
    """

    # Check the dimensions.
    assert( mask.shape[0] == img.shape[0] )
    assert( mask.shape[1] == img.shape[1] )

    # Get the dimensions
    height = mask.shape[0]
    width  = mask.shape[1]

    # Transfer memory to the CUDA device.
    dMask = cuda.to_device(mask)
    dImg  = cuda.to_device(img)

    # Filter.
    cuda.synchronize()
    k_filter_horizontal[[1, int(height/16)+1, 1], [1, 16, 1]](dMask, dImg)
    k_filter_vertical[[int(width/16)+1, 1, 1], [16, 1, 1]](dMask, dImg)
    cuda.synchronize()

    # Transfer memory to the host.
    mask =dMask.copy_to_host()
    img = dImg.copy_to_host()

    return mask, img

def get_parts(fn):
    parts0 = os.path.split(fn)
    parts1 = os.path.splitext(parts0[1])

    return parts0[0], parts1[0], parts1[1]

def write_float_image(fn, img):
    """
    Only supports PNG file.
    """

    # Make a deep copy.
    im = copy.deepcopy(img)

    # Normalize.
    im = im - im.min()
    im = ( im / im.max() ) * 255
    im = im.astype(np.uint8)

    # Save the image.
    cv2.imwrite( fn, im, [cv2.IMWRITE_PNG_COMPRESSION, 0] )

if __name__ == "__main__":
    print("Local test Filter.py")

    INPUT_MASK = "/home/yaoyu/Projects/DeepStereoAssistant/WD/PU_03_IF_02_Middlebury/TestResults/Bounds_Middlebury_00/Disparity_Occ.npy"
    INPUT_IMG  = "/home/yaoyu/Projects/DeepStereoAssistant/WD/PU_03_IF_02_Middlebury/TestResults/Bounds_Middlebury_00/Disparity_OccMasked.npy"

    # Load the mask and the image.
    mask = np.load(INPUT_MASK)
    img  = np.load(INPUT_IMG)

    maskFiltered, imgFiltered = guided_filter_3(mask, img)

    # Save the filtered mask and image back to the input folder.
    parts = get_parts(INPUT_MASK)
    outFn = "%s/%s_Filtered%s" % ( parts[0], parts[1], parts[2] )
    np.save( outFn, maskFiltered )
    outFn = "%s/%s_Filtered.png" % ( parts[0], parts[1] )
    write_float_image(outFn, maskFiltered)

    parts = get_parts(INPUT_IMG)
    outFn = "%s/%s_Filtered%s" % ( parts[0], parts[1], parts[2] )
    np.save( outFn, imgFiltered )
    outFn = "%s/%s_Filtered.png" % ( parts[0], parts[1] )
    write_float_image(outFn, imgFiltered)
