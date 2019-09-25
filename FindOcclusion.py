from __future__ import print_function

import argparse
import copy
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

nbVersionMin0 = 0
nbVersionMin1 = 45
from numba import __version__ as nbVersion
nbVersionParts = nbVersion.split(".")
if ( int(nbVersionParts[0]) <= nbVersionMin0 and int(nbVersionParts[1]) < nbVersionMin1 ):
    raise Exception("Minimum %d.%d of numba is required. The current version is %s." % \
        (nbVersionMin0, nbVersionMin1, nbVersion))

from numba import cuda
from Filter import guided_filter_3

INVALID_DISP = -1

def get_parts(fn):
    parts0 = os.path.split(fn)
    parts1 = os.path.splitext(parts0[1])

    return parts0[0], parts1[0], parts1[1]

def find_occlusion(ref, threshold=2, invalid=1):
    if ( 2 != len(ref.shape) ):
        raise Exception("ref.shape = {}".format(ref.shape))

    # Create a dummy tst image.
    tst = np.zeros_like(ref, dtype=np.int64)
    occ = np.zeros_like(ref, dtype=np.uint8)

    for i in range( ref.shape[0] ):
        for j in range( ref.shape[1] ):
            # Get the disparity.
            d = int( np.rint(ref[i, j]) )

            if ( d <= invalid ):
                continue

            jT = j - d

            if ( jT < 0 ):
                continue

            # Check the position in the tst image.
            if ( tst[i, jT] == 0 ):
                # No pixel claim a match here.
                tst[i, jT] = j
            else:
                # Other pixel claims a match here.

                if ( j - tst[i, jT] > threshold ):
                    # Current pixel occludes the other pixel.
                    occ[i, tst[i, jT]] = 255
                    # Update tst.
                    tst[i, jT] = j
                elif ( j - tst[i, jT] >=1 ):
                    pass
                else:
                    # Could not happen!
                    raise Exception("i: %d, j: %d, d: %d, jT: %d, tst[i, jT]: %d. " % \
                        (i, j, d, jT, tst[i, jT]) )

    # Return the mask.
    return occ

@cuda.jit
def k_find_occlusion(ref, occ, buffer, threshold, invalid):
    ty = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y

    # Check if the current thread is out of the region.
    if ( ty >= ref.shape[0] ):
        return

    # Process the current line.
    for j in range( ref.shape[1] ):
        # Get the disparity.
        d = int( round(ref[ty, j]) )

        if ( d <= invalid ):
            continue

        jT = j - d

        if ( jT < 0 ):
            continue

        # Check the position in the tst image.
        if ( buffer[ty, jT] == 0 ):
            # No pixel claim a match here.
            buffer[ty, jT] = j
        else:
            # Other pixel claims a match here.

            if ( j - buffer[ty, jT] > threshold ):
                # Current pixel occludes the other pixel.
                occ[ty, buffer[ty, jT]] = 255
                # Update tst.
                buffer[ty, jT] = j
            elif ( j - buffer[ty, jT] >=1 ):
                pass
            else:
                # Could not happen!
                raise Exception("You promised that it could not happen! So what!?")

def occ_proposal(disp, flagCUDA=True):
    # Find occlusion.
    if ( flagCUDA ):
        # Create a dummy tst image.
        buffer = np.zeros_like(disp, dtype=np.int64)
        occ    = np.zeros_like(disp, dtype=np.uint8)

        # Transfer memory to the device.
        dDisp   = cuda.to_device(disp)
        dBuffer = cuda.to_device(buffer)
        dOcc    = cuda.to_device(occ)

        # Invoke CUDA kernel.
        cuda.synchronize()
        k_find_occlusion[[1, int(disp.shape[0]/16+1), 1], [1, 16, 1]]( dDisp, dOcc, dBuffer, 2, 1 )
        cuda.synchronize()

        # Transfer memory to the host.
        occ = dOcc.copy_to_host()
    else:
        occ = find_occlusion(disp)

    return occ

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
    parser = argparse.ArgumentParser(description="Plot a 2D heat map of dispairty error.")

    parser.add_argument("inputdir", type=str, \
        help="The input directory.")
    parser.add_argument("pattern", type=str, \
        help="The search pattern for the input .npy files.")
    parser.add_argument("--output-dir", type=str, default="", \
        help="The output directory. If left blank, the input directory will be used.")
    parser.add_argument("--sub", action="store_true", default=False, \
        help="Set this flag to let the script search sub-folders recursively. The result files will also be stored in sub-folders.")
    parser.add_argument("--cuda", action="store_true", default=False, \
        help="Use CUDA acceleration.")
    parser.add_argument("--only-list", action="store_true", default=False, \
        help="Set this to by pass the processing and only show the found input files.")
    
    args = parser.parse_args()

    # Find all the images.
    if ( args.sub ):
        files = sorted( glob.glob( args.inputdir + "/**/" + args.pattern, recursive=True ) )
    else:
        files = sorted( glob.glob( args.inputdir + "/" + args.pattern, recursive=False ) )

    if ( 0 == len(files) ):
        raise Exception("No file found with %s/%s." % ( args.inputdir, args.pattern ))

    # Test the output directory.
    if ( "" == args.output_dir ):
        outDir = args.inputdir
    else:
        outDir = args.output_dir

    if ( not os.path.isdir(outDir) ):
        os.makedirs(outDir)
    
    nCharInputDir = len( args.inputdir )
    if ( args.inputdir[-1] != "/" ):
        nCharInputDir += 1

    for f in files:
        print(">>> ")
        print(f)

        if ( args.only_list ):
            continue

        # Read the disparity file.
        disp = np.load(f)

        # Find occlusion.
        occ = occ_proposal(disp, args.cuda)

        # Filter occ with a median filter.
        # occ = cv2.medianBlur(occ, 3)
        # occ = cv2.medianBlur(occ, 3)

        # Make a deep copy of the disparity map and the occlusion map.
        dispBackup = copy.deepcopy( disp )
        occBackup = copy.deepcopy( occ )

        # Filter the occlusion.
        occ, disp = guided_filter_3(occ, disp)

        # Create the masked version of the disparity map.
        dispMasked = copy.deepcopy(disp)
        mask = occ == 255
        dispMasked[mask] = INVALID_DISP

        # Save the masked disparity map and the mask.
        parts = get_parts(f)

        if ( args.sub ):
            subDir = parts[0][nCharInputDir:]
            outFnDisp      = "%s/%s/%s_DispFT.npy" % ( outDir, subDir, parts[1] )
            outFnDispImg   = "%s/%s/%s_DispFT.png" % ( outDir, subDir, parts[1] )
            outFnDispBK    = "%s/%s/%s_DispBK.npy" % ( outDir, subDir, parts[1] )
            outFnDispBKImg = "%s/%s/%s_DispBK.png" % ( outDir, subDir, parts[1] )
            outFnDispOC    = "%s/%s/%s_DispOC.npy" % ( outDir, subDir, parts[1] )
            outFnDispOCImg = "%s/%s/%s_DispOC.png" % ( outDir, subDir, parts[1] )
            outFnOcc       = "%s/%s/%s_Occ.npy" % ( outDir, subDir, parts[1] )
            outFnOccImg    = "%s/%s/%s_Occ.png" % ( outDir, subDir, parts[1] )
            outFnOccBK     = "%s/%s/%s_OccBK.npy" % ( outDir, subDir, parts[1] )
            outFnOccBKImg  = "%s/%s/%s_OccBK.png" % ( outDir, subDir, parts[1] )
        else:
            outFnDisp      = "%s/%s_DispFT.npy" % ( outDir, parts[1] )
            outFnDispImg   = "%s/%s_DispFT.png" % ( outDir, parts[1] )
            outFnDispBK    = "%s/%s_DispBK.npy" % ( outDir, parts[1] )
            outFnDispBKImg = "%s/%s_DispBK.png" % ( outDir, parts[1] )
            outFnDispOC    = "%s/%s_DispOC.npy" % ( outDir, parts[1] )
            outFnDispOCImg = "%s/%s_DispOC.png" % ( outDir, parts[1] )
            outFnOcc       = "%s/%s_Occ.npy" % ( outDir, parts[1] )
            outFnOccImg    = "%s/%s_Occ.png" % ( outDir, parts[1] )
            outFnOccBK     = "%s/%s_OccBK.npy" % ( outDir, parts[1] )
            outFnOccBKImg  = "%s/%s_OccBK.png" % ( outDir, parts[1] )
        
        print(outFnDisp)
        np.save(outFnDisp, disp)
        np.save(outFnDispBK, dispBackup)
        np.save(outFnDispOC, dispMasked)
        np.save(outFnOcc, occ)
        np.save(outFnOccBK, occBackup)

        write_float_image(outFnDispImg, disp)
        write_float_image(outFnDispBKImg, dispBackup)
        write_float_image(outFnDispOCImg, dispMasked)

        # Make the occlusion map have the same color of the Middlebury ground truth.
        mask = occ == 255
        occ[mask] = 128
        mask = occ != 128
        occ[mask] = 255

        mask = occBackup == 255
        occBackup[mask] = 128
        mask = occBackup != 128
        occBackup[mask] = 255

        cv2.imwrite(outFnOccImg, occ, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(outFnOccBKImg, occBackup, [cv2.IMWRITE_PNG_COMPRESSION, 0])
