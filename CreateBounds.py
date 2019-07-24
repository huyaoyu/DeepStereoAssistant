from __future__ import print_function

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

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
    
    args = parser.parse_args()

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

    # Save the disparity and sigma.
    dispFn = args.out_dir + "/" + args.out_name_disp + ".npy"
    np.save(dispFn, dispR)

    sigFn = args.out_dir + "/" + args.out_name_sig + ".npy"
    np.save(sigFn, sigR)

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

    print("Done.")
