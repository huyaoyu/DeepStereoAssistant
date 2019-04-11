from __future__ import print_function

import argparse
import glob
import os

def list_files_sample(dataPath):
    if ( False == os.path.isdir(dataPath) ):
        raise Exception("%s does not exist." % (dataPath))

    allImgL = sorted( glob.glob(dataPath + "/RGB_cleanpass/left/*.png") )
    allImgR = sorted( glob.glob(dataPath + "/RGB_cleanpass/right/*.png") )
    allDisp = sorted( glob.glob(dataPath + "/disparity/*.pfm") )

    nImgL = len( allImgL )
    nImgR = len( allImgR )
    nDisp = len( allDisp )

    if ( nImgL != nImgR or nImgL != nDisp ):
        raise Exception("In consistent file numbers. nImgL = %d, nImgR = %d, nDisp = %d." % ( nImgL, nImgR, nDisp ))

    #  trainImgL, trainImgR, trainDisp, testImgL, testImgR, testDisp
    return allImgL, allImgR, allDisp, allImgL, allImgR, allDisp

def list_files_sceneflow_FlyingThings(rootPath):
    """
    rootPath: The path of the root of the dataset. The directory contains "frames_cleanpass" and "disparity" folders.
    """

    if ( False == os.path.isdir(rootPath + "/frames_cleanpass") ):
        raise Exception("%s does not exist." % ( rootPath + "/frames_cleanpass" ))

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

def list_files_sceneflow_Monkaa(rootPath, withoutRootPath=False):
    """
    rootPath: The path of the root of the dataset. The directory contains "frames_cleanpass" and "disparity" folders.

    Difference with the flying things dataset it that Monkaa does not have separated TRAIN and TEST folders.
    """

    if ( False == os.path.isdir(rootPath + "/frames_cleanpass") ):
        raise Exception("%s does not exist." % ( rootPath + "/frames_cleanpass" ))

    import ipdb; ipdb.set_trace()

    # Search the "frames_cleanpass" directory recursively.
    allImgL = sorted( glob.glob( rootPath + "/frames_cleanpass/**/left/*.png", recursive=True ) )

    if ( True == withoutRootPath ):
        lenRootPath = len(rootPath)
        nImgs = len(allImgL)

        for i in range(nImgs):
            allImgL[i] = allImgL[i][lenRootPath+1:]

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

    #      leftTrain, rightTrian, dispTrain, leftTest, rightTest, dispTest.
    return   allImgL,    allImgR,   allDisp,       [],        [],       []

def write_string_list_2_file(fn, s):
    """
    Write a string list s into a file fn.
    Every element of s will be written as a separate line.
    """

    with open(fn, "w") as fp:
        n = len(s)

        for i in range(n):
            temp = "%s\n" % (s[i].strip())
            fp.write(temp)

def read_string_list(fn):
    """
    Read a file contains lines of strings. A list will be returned.
    Each element of the list contains a single entry of the input file.
    Leading and tailing white spaces, tailing carriage return will be stripped.
    """

    if ( False == os.path.isfile( fn ) ):
        raise Exception("%s does not exist." % (fn))
    
    with open(fn, "r") as fp:
        lines = fp.read().splitlines()

        n = len(lines)

        for i in range(n):
            lines[i] = lines[i].strip()

    return lines

USAEG_MSG = """
Generate input files files. 

Six files will be generated: 
(1) Training reference image / left image list, InputImageTrainingL.txt.
(2) Training target image / right image list, InputImageTrainingR.txt.
(3) Training referece / left disparity list, InputDisparityTraining.txt.
(4) Testing reference image / left image list, InputImageTestingL.txt.
(5) Testing target image / right image list, InputImageTestingR.txt.
(6) Testing referece / left disparity list, InputDisparityTesting.txt.

All six files will be written to the directory specified by --output-dir argument. 
If the output directory does not exist, this script will try to create the directory.

"""

if __name__ == "__main__":
    print("Generate input files files.")

    parser = argparse.ArgumentParser(description=USAEG_MSG)

    parser.add_argument("--flying-things-root-dir", type=str, default="", \
        help="The root directory of the flying things dataset.")

    parser.add_argument("--monkaa-root-dir", type=str, default="", \
        help="The root directory of the Monkaa dataset.")
    
    parser.add_argument("--output-dir", type=str, default="./", \
        help="The output directory of the file. There will be six files.")

    args = parser.parse_args()

    # Get the lists for the Flyingthings dataset.
    if ( "" != args.flying_things_root_dir ):
        imgTrainL_FT, imgTrainR_FT, dispTrain_FT, imgTestL_FT, imgTestR_FT, dispTest_FT = \
            list_files_sceneflow_FlyingThings( args.flying_things_root_dir )
    else:
        imgTrainL_FT, imgTrainR_FT, dispTrain_FT, imgTestL_FT, imgTestR_FT, dispTest_FT = \
            [], [], [], [], [], []
    
    # Get the lists for the Monkka dataset.
    if ( "" != args.monkaa_root_dir ):
        imgTrainL_MK, imgTrainR_MK, dispTrain_MK, imgTestL_MK, imgTestR_MK, dispTest_MK = \
            list_files_sceneflow_Monkaa( args.monkaa_root_dir )
    else:
        mgTrainL_MK, imgTrainR_MK, dispTrain_MK, imgTestL_MK, imgTestR_MK, dispTest_MK = \
            [], [], [], [], [], []

    # Merge the lists.
    imgTrainL = imgTrainL_FT + imgTrainL_MK
    imgTrainR = imgTrainR_FT + imgTrainR_MK
    dispTrain = dispTrain_FT + dispTrain_MK
    imgTestL  = imgTestL_FT + imgTestL_MK
    imgTestR  = imgTestR_FT + imgTestR_MK
    dispTest  = dispTest_FT + dispTest_MK

    nImgTrainL, nImgTrainR, nDispTrain, nImgTestL, nImgTestR, nDispTest = \
        len(imgTrainL), len(imgTrainR), len(dispTrain), len(imgTestL), len(imgTestR), len(dispTest)

    if ( 0 == nImgTrainL or 0 == nImgTrainR or 0 == nDispTrain or \
         0 == nImgTestL or 0 == nImgTestR or 0 == nDispTest ):
        raise Exception("Zero length list found. Train: %d, %d, %d. Test: %d, %d, %d." % \
            ( nImgTrainL, nImgTrainR, nDispTrain, nImgTestL, nImgTestR, nDispTest ))
    else:
        print("Train: %d, %d, %d. Test: %d, %d, %d." % \
            ( nImgTrainL, nImgTrainR, nDispTrain, nImgTestL, nImgTestR, nDispTest ))

    if ( False == os.path.isdir(args.output_dir) ):
        os.makedirs( args.output_dir )

    # Save the files.
    write_string_list_2_file( args.output_dir + "/InputImageTrainL.txt", imgTrainL )
    write_string_list_2_file( args.output_dir + "/InputImageTrainR.txt", imgTrainR )
    write_string_list_2_file( args.output_dir + "/InputDisparityTrain.txt", dispTrain )
    write_string_list_2_file( args.output_dir + "/InputImageTestL.txt", imgTestL )
    write_string_list_2_file( args.output_dir + "/InputImageTestR.txt", imgTestR )
    write_string_list_2_file( args.output_dir + "/InputDisparityTest.txt", dispTest )

    print("Files are written. All done.")
