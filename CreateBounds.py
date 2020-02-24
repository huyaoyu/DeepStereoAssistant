from __future__ import print_function

import argparse
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import IO

# DISP_INVALID = -1

def convert_invalid_value_argument(arg):
    if ( "inf" == arg ):
        return np.inf
    else:
        return float(arg.strip())

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

def main(args):
    print(args.disp)

    # Convert the invalid value argument.
    DISP_INVALID = convert_invalid_value_argument(args.invalid_value)

    # Load the input files.
    disp = np.load(args.disp).astype(np.float32)
    sig  = np.load(args.sig).astype(np.float32)

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

    # Create the output directory.
    if ( not os.path.isdir( args.out_dir ) ):
        os.makedirs( args.out_dir )

    # Save the disparity as pfm file.
    dispPFM = copy.deepcopy( dispR )
    if ( args.n_invalid > 0 ):
        if ( args.n_invalid > dispR.shape[1] ):
            raise Exception("Cannot make %d columns as invalid. Image width is %d." % ( args.n_invalid, dispR.shape[1] ))

        dispPFM[:, :args.n_invalid] = DISP_INVALID
    
    outFn = "%s/%s.pfm" % ( args.out_dir, args.out_name_disp )
    IO.writePFM(outFn, dispPFM)

    # Save the disparity and sigma.
    dispFn = args.out_dir + "/" + args.out_name_disp + ".npy"
    np.save(dispFn, dispR)

    dispImgFn = args.out_dir + "/" + args.out_name_disp_img + ".png"
    save_float_image_normalized( dispImgFn, dispR )

    sigFn = args.out_dir + "/" + args.out_name_sig + ".npy"
    np.save(sigFn, sigR)

    sigImgFn = args.out_dir + "/" + args.out_name_sig_img + ".png"
    save_float_image_normalized( sigImgFn, sigR )

    sigFigFn = args.out_dir + "/" + args.out_name_sig + ".png"
    fig = plt.figure()
    plt.imshow(sigR)
    fig.savefig(sigFigFn)
    
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
    lowerBoundFn = args.out_dir + "/" + args.out_name_lower + ".npy"
    np.save(lowerBoundFn, lowerBound)

    upperBoundFn = args.out_dir + "/" + args.out_name_upper + ".npy"
    np.save(upperBoundFn, upperBound)

    # Draw the images of the bounds.
    lowerBoundFigFn = args.out_dir + "/" + args.out_name_lower + ".png"
    fig = plt.figure()
    plt.imshow(lowerBound)
    fig.savefig(lowerBoundFigFn)

    upperBoundFigFn = args.out_dir + "/" + args.out_name_upper + ".png"
    fig = plt.figure()
    plt.imshow(upperBound)
    fig.savefig(upperBoundFigFn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the lower and upper bounds file.")

    parser.add_argument("--disp", type=str, \
        help="The disparity file.")
    parser.add_argument("--sig", type=str, \
        help="The sigma file.")
    parser.add_argument("--n-sig", type=float, default=1.0, \
        help="The number of sigmas.")
    parser.add_argument("--height", type=int, \
        help="The height of the output image.")
    parser.add_argument("--width", type=int, \
        help="The the width of the output image.")
    parser.add_argument("--out-dir", type=str, default="./", \
        help="The output directory.")
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
    parser.add_argument("--invalid-value", type=str, default="-1", \
        help="The invalid disparity value. Use numbers. Use inf to set numpy infinity.")
    
    args = parser.parse_args()

    main(args)

    print("Done.")
