from __future__ import print_function

import argparse
import numpy as np
import os

from PointCloud.PLYHelper import write_PLY

Q_FLIP = np.array( [ \
    [ 1.0,  0.0,  0.0, 0.0 ], \
    [ 0.0, -1.0,  0.0, 0.0 ], \
    [ 0.0,  0.0, -1.0, 0.0 ], \
    [ 0.0,  0.0,  0.0, 1.0 ] \
         ], dtype=np.float32 )

def convert_RGB_string(s):
    """
    Convert a string of RGB values separated by comma into a list.
    The values in the list are clipped in the range of [0, 255].
    """

    s = s.split(",")

    RGB = []

    for n in s:
        n = n.strip()

        RGB.append( int(n) )

        if ( RGB[-1] > 255 ):
            RGB[-1] = 255
        
        if ( RGB[-1] < 0 ):
            RGB[-1] = 0
    
    return RGB

def create_color_image(c, h, w):
    """
    Create a 3-channel color image as NumPy array. The dtype will be uint8.
    c: A 3-element list contains the RGB values.
    h: Height of the output image.
    w: Width of the output image.
    """

    img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(3):
        img[ :, :, i ] = c[i]

    return img

if __name__ == "__main__":
    print("========== Generate disparity range file. ==========")

    # Arguments.
    parser = argparse.ArgumentParser(description="Generate disparity range files.")

    parser.add_argument("--disp", type=str, \
        help="The disparity file.")
    parser.add_argument("--sigma", type=str, \
        help="The sigma file.")
    parser.add_argument("--out-dir", type=str, \
        help="The output directory")
    parser.add_argument("--max", type=int, default=255, \
        help="The maximum disparity value.")
    parser.add_argument("--n", type=int, default=3, \
        help="Number of sigmas to apply on the disparity ranges.")
    parser.add_argument("--ascii", action="store_true", default=False, \
        help="Store the ranges in ASCII format.")
    parser.add_argument("--ply", action="store_true", default=False, \
        help="Save PLY files as well.")
    parser.add_argument("--ply-ascii", action="store_true", default=False, \
        help="Store the PLY files in ASCII format.")
    parser.add_argument("--Q", type=str, default="", \
        help="The Q matrix file. Must be provided is --ply is issued.")
    parser.add_argument("--Q-flip", action="store_true", default=False, \
        help="Flip the Q matrix along the x-axis.")
    parser.add_argument("--ply-color-lower", type=str, default="0, 0, 255", \
        help="The color of the lower range points in the PLY file. The values are in RGB order.")
    parser.add_argument("--ply-color-upper", type=str, default="255, 0, 0", \
        help="The color of the lower range points in the PLY file. The values are in RGB order.")

    args = parser.parse_args()

    if ( True == args.ply and 0 == len( args.Q )):
        raise Exception("Q must be specified if --ply is issued.")

    # Check the output directory.
    if ( not os.path.isdir( args.out_dir ) ):
        os.makedirs( args.out_dir )

    # Load the disparity and sigma files.
    disp  = np.load(args.disp).astype(np.float32)
    sigma = np.load(args.sigma).astype(np.float32)

    # The upper and lower ranges.
    dispUpper = disp + args.n * sigma
    dispUpper = np.clip( dispUpper, 0, args.max )

    dispLower = disp - args.n * sigma
    dispLower = np.clip( dispLower, 0, args.max )

    # Figure out the filename.
    stemFn = os.path.splitext( os.path.split( args.disp )[1] )[0]
    baseFn = args.out_dir + "/" + stemFn

    # Save the ranges.
    if ( False == args.ascii ):
        np.save( baseFn + "_Lower.npy", dispLower )
        np.save( baseFn + "_Upper.npy", dispUpper )

        print("Save %s." % (baseFn + "_Lower.npy"))
        print("Save %s." % (baseFn + "_Upper.npy"))
    else:
        np.savetxt( baseFn + "_Lower.dat", dispLower )
        np.savetxt( baseFn + "_Upper.dat", dispUpper )

        print("Save %s in ASCII format." % (baseFn + "_Lower.npy"))
        print("Save %s in ASCII format." % (baseFn + "_Upper.npy"))

    # Save the PLY files.
    if ( True == args.ply ):
        print("Save PLY files as well.")

        # Load the Q matrix.
        Q = np.loadtxt( args.Q, dtype=np.float32 )

        if ( True == args.Q_flip ):
            Q = Q_FLIP.dot( Q )

        # Prepare the color image.
        colorLower = convert_RGB_string( args.ply_color_lower )
        colorUpper = convert_RGB_string( args.ply_color_upper )

        print("colorLower = {}.".format( colorLower ))
        print("colorUpper = {}.".format( colorUpper ))

        imgColorLower = create_color_image( colorLower, dispLower.shape[0], dispLower.shape[1] )
        imgColorUpper = create_color_image( colorUpper, dispUpper.shape[0], dispUpper.shape[1] )

        if ( False == args.ply_ascii ):
            write_PLY( baseFn + "_Lower.ply", dispLower, Q, color=imgColorLower, binary=True )
            write_PLY( baseFn + "_Upper.ply", dispUpper, Q, color=imgColorUpper, binary=True )

            print("Save %s in binary format." % ( baseFn + "_Lower.ply" ))
            print("Save %s in binary format." % ( baseFn + "_Upper.ply" ))
        else:
            write_PLY( baseFn + "_Lower.ply", dispLower, Q, color=imgColorLower, binary=False )
            write_PLY( baseFn + "_Lower.ply", dispUpper, Q, color=imgColorUpper, binary=False )

            print("Save %s in ASCII format." % ( baseFn + "_Lower.ply" ))
            print("Save %s in ASCII format." % ( baseFn + "_Upper.ply" ))
        
    print("========== Done. ==========")
