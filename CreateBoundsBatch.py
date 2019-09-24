from __future__ import print_function

import argparse
import copy
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import IO

DISP_INVALID = -1

def get_parts(fn):
    parts0 = os.path.split(fn)
    parts1 = os.path.splitext(parts0[1])

    return parts0[0], parts1[0], parts1[1]

def save_float_image_normalized(fn, img, lowerBound=None, upperBound=None):
    """
    Save a float image as a image file.
    fn: The output file name.
    img: The input NumPy array.
    lowerBound: The lower bound for clipping and normalization.
    upperBound: The upper bound for clipping and normalization.

    Set lowerBound or upperBound to None to use the minimum and maximum values
    as the bounds for normalization.

    NOTE: Only works with single channel image.
    """

    if ( 2 != len( img.shape ) ):
        raise Exception("Only supports single channel image. img.shape = {}".format(img.shape))
    
    if ( lowerBound is not None and upperBound is not None ):
        if ( lowerBound >= upperBound ):
            raise Exception("Wrong bounds. [%f, %f]" % (lowerBound, upperBound))
    
    # Clip and normalize.
    img = copy.deepcopy( img )

    if ( lowerBound is None or upperBound is None ):
        lowerBound = img.min()
        upperBound = img.max()

    img = np.clip( img, lowerBound, upperBound )
    img = img - img.min()
    img = img / img.max()

    imgInt = img*255
    imgInt = imgInt.astype(np.uint8)

    # Save the image.
    cv2.imwrite(fn, imgInt, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the lower and upper bounds file.")

    parser.add_argument("input_dir", type=str, \
        help="The intpu directory.")
    parser.add_argument("disp", type=str, \
        help="The search pattern of the disparity file.")
    parser.add_argument("sig", type=str, \
        help="The search pattern of the sigma file.")
    parser.add_argument("--sub", action="store_true", default=False, \
        help="Set this flag to let the script search sub-folders recursively. The result files will also be stored in sub-folders.")
    parser.add_argument("--n-sig", type=float, default=1.0, \
        help="The number of sigmas.")
    parser.add_argument("--height", type=int, \
        help="The height of the output image.")
    parser.add_argument("--width", type=int, \
        help="The the width of the output image.")
    parser.add_argument("--out-name-lower", type=str, default="LowerBound", \
        help="The base file name of the lower bound file.")
    parser.add_argument("--out-name-upper", type=str, default="UpperBound", \
        help="The base file name of the upper bound file.")
    parser.add_argument("--out-name-disp", type=str, default="Disparity", \
        help="The base file name of the scaled disparity file.")
    parser.add_argument("--out-name-sig", type=str, default="Sigma", \
        help="The base file name of the scaled sigma file.")
    parser.add_argument("--out-name-disp-img", type=str, default="Disparity", \
        help="The base file name of the disparity image.")
    parser.add_argument("--out-name-sig-img", type=str, default="SigmaGray", \
        help="The base file name of the scaled sigma image.")
    parser.add_argument("--n-invalid", type=int, default=0, \
        help="Number of pixel columns to assign the 'invalid' value. Starting from the left border of the disparity map. This only affects the pfm file.")
    parser.add_argument("--only-list", action="store_true", default=False, \
        help="Set this flag to show the input file names only.")
    
    args = parser.parse_args()

    # Find all the input files.
    if ( args.sub ):
        disps = sorted( glob.glob( args.input_dir + "/**/" + args.disp, recursive=True ) )
        sigs  = sorted( glob.glob( args.input_dir + "/**/" + args.sig,  recursive=True ) )
    else:
        disps = sorted( glob.glob( args.input_dir + "/" + args.disp, recursive=False ) )
        sigs  = sorted( glob.glob( args.input_dir + "/" + args.sig,  recursive=False ) )

    if ( 0 == len(disps) ):
        raise Exception("No file found with %s + %s." % ( args.inputdir, args.disp ))

    if ( 0 == len(sigs) ):
        raise Exception("No file found with %s + %s." % ( args.inputdir, args.sig ))

    if ( len(disps) != len(sigs) ):
        raise Exception("Numbers of disparity files (%d) and sigma files (%d) are not the same." % \
            ( len(disps), len(sigs) ))

    for i in range(len(disps)):
        dispFn = disps[i]
        sigFn  = sigs[i]

        partsDisp = get_parts(dispFn)
        partsSig  = get_parts(sigFn)

        # The output directory.
        outDir = partsDisp[0]

        print(dispFn)

        if ( partsDisp[0] != partsSig[0] ):
            raise Exception("i = %d, \npartsDisp[0] = %s, \npartsSig[0] = %s" % \
                ( i, partsDisp[0], partsSig[0] ))

        if ( args.only_list ):
            print(outDir)
            print("==========")
            continue

        # Load the input files.
        disp = np.load(dispFn).astype(np.float32)
        sig  = np.load(sigFn).astype(np.float32)

        # Resize the files.
        dispR = cv2.resize( disp, ( args.width, args.height ), interpolation=cv2.INTER_LINEAR )
        sigR  = cv2.resize( sig,  ( args.width, args.height ), interpolation=cv2.INTER_LINEAR )

        # The width resizing factor.
        f = args.width / disp.shape[1]
        print("f = %f." % (f))

        dispR = dispR * f
        sigR  = sigR  * f

        # Show some information about the sigma.
        print("sigma: (%f, %f)." % ( sigR.min(), sigR.max() ))

        # Save the disparity as pfm file.
        dispPFM = copy.deepcopy( dispR )
        if ( args.n_invalid > 0 ):
            if ( args.n_invalid > dispR.shape[1] ):
                raise Exception("Cannot make %d columns as invalid. Image width is %d." % ( args.n_invalid, dispR.shape[1] ))

            dispPFM[:, :args.n_invalid] = DISP_INVALID
        
        outFn = "%s/%s_i%d.pfm" % ( outDir, args.out_name_disp, args.n_invalid )
        IO.writePFM(outFn, dispPFM)

        # Save the disparity and sigma.
        dispFn = outDir + "/" + args.out_name_disp + ".npy"
        np.save(dispFn, dispR)

        dispImgFn = outDir + "/" + args.out_name_disp_img + ".png"
        save_float_image_normalized( dispImgFn, dispR )

        sigFn = outDir + "/" + args.out_name_sig + ".npy"
        np.save(sigFn, sigR)

        sigImgFn = outDir + "/" + args.out_name_sig_img + ".png"
        save_float_image_normalized( sigImgFn, sigR )

        sigFigFn = outDir + "/" + args.out_name_sig + ".png"
        fig = plt.figure()
        plt.imshow(sigR)
        fig.savefig(sigFigFn)
        plt.close(fig)
        
        # The lower and upper bounds.
        lowerBound = dispR - args.n_sig * sigR
        upperBound = dispR + args.n_sig * sigR

        lowerBound = lowerBound.astype(np.int32)
        upperBound = upperBound.astype(np.int32)

        # Clip the lowerBound.
        lowerBound = np.clip( lowerBound, 1, lowerBound.max() ).astype(np.int32)

        # Show some information about the bounds.
        print("Lower bound: (%d, %d)." % ( lowerBound.min(), lowerBound.max() ))
        print("Upper bound: (%d, %d)." % ( upperBound.min(), upperBound.max() ))

        # Save the lower and upper bounds.
        lowerBoundFn = outDir + "/" + args.out_name_lower + ".npy"
        np.save(lowerBoundFn, lowerBound)

        upperBoundFn = outDir + "/" + args.out_name_upper + ".npy"
        np.save(upperBoundFn, upperBound)

        # Draw the images of the bounds.
        lowerBoundFigFn = outDir + "/" + args.out_name_lower + ".png"
        fig = plt.figure()
        plt.imshow(lowerBound)
        fig.savefig(lowerBoundFigFn)
        plt.close(fig)

        upperBoundFigFn = outDir + "/" + args.out_name_upper + ".png"
        fig = plt.figure()
        plt.imshow(upperBound)
        fig.savefig(upperBoundFigFn)
        plt.close(fig)

        print("==========")

    print("Done.")
