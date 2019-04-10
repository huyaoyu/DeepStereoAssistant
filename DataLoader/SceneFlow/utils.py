from __future__ import print_function

import glob
import os

def list_files_sample(dataPath):
    if ( False == os.path.isdir(dataPath) ):
        Exception("%s does not exist." % (dataPath))

    allImgL = sorted( glob.glob(dataPath + "/RGB_cleanpass/left/*.png") )
    allImgR = sorted( glob.glob(dataPath + "/RGB_cleanpass/right/*.png") )
    allDisp = sorted( glob.glob(dataPath + "/disparity/*.pfm") )

    nImgL = len( allImgL )
    nImgR = len( allImgR )
    nDisp = len( allDisp )

    if ( nImgL != nImgR or nImgL != nDisp ):
        Exception("In consistent file numbers. nImgL = %d, nImgR = %d, nDisp = %d." % ( nImgL, nImgR, nDisp ))

    #  trainImgL, trainImgR, trainDisp, testImgL, testImgR, testDisp
    return allImgL, allImgR, allDisp, allImgL, allImgR, allDisp

def list_files_sceneflow_FlyingThings(rootPath):
    """
    rootPath: The path of the root of the dataset. The directory contains "frames_cleanpass" and "disparity" folders.
    """

    if ( False == os.path.isdir(rootPath) ):
        Exception("%s does not exist." % ( rootPath ))

    # Search the "frames_cleanpass/TRAIN" directory recursively.
    allImgL = sorted( glob.glob( rootPath + "/frames_cleanpass/TRAIN/**/left/*.png", recursive=True ) )

    # Generate all filenames assuming they are all exist on the filesystem.
    allImgR = []
    allDisp = []

    for fn in allImgL:
        # Make the path for the right image.
        fnR = fn.replace( "left", "right" )
        allImgR.append( fnR )

        # Make the path for the disparity file.
        fnD = fn.replace( "frames_cleanpass", "disparity" )
        fnD = fnD.replace( ".png", ".pfm" )
        allDisp.append( fnD )

    # Search the "frames_cleanpass/TEST" directory recursively.
    allTestImgL = sorted( glob.glob( rootPath + "/frames_cleanpass/TEST/**/left/*.png", recursive=True ) )

    # Generate all filenames assuming they are all exist on the filesystem.
    allTestImgR = []
    allTestDisp = []

    for fn in allTestImgL:
        # Make the path for the right image.
        fnR = fn.replace( "left", "right" )
        allTestImgR.append( fnR )

        # Make the path for the disparity file.
        fnD = fn.replace( "frames_cleanpass", "disparity" )
        fnD = fnD.replace( ".png", ".pfm" )
        allTestDisp.append( fnD )

    return allImgL, allImgR, allDisp, allTestImgL, allTestImgR, allTestDisp