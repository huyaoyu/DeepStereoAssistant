
from __future__ import print_function

import copy
import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..DatasetUtils import find_filenames_recursively

def show_sample(sample):
    # Create a matplotlib figure.
    fig = plt.figure()

    axImgL = plt.subplot(2, 2, 1)
    plt.tight_layout()
    axImgL.set_title("image0")
    axImgL.axis("off")
    plt.imshow( sample["image0"] )

    axImgR = plt.subplot(2, 2, 2)
    plt.tight_layout()
    axImgR.set_title("image1")
    axImgR.axis("off")
    plt.imshow( sample["image1"] )

    axDispL = plt.subplot(2, 2, 3)
    plt.tight_layout()
    axDispL.set_title("disp0")
    axDispL.axis("off")
    plt.imshow( sample["disparity0"] )

    axDispR = plt.subplot(2, 2, 4)
    plt.tight_layout()
    axDispR.set_title("disp1")
    axDispR.axis("off")
    plt.imshow( sample["disparity1"] )

    plt.show()

def show_sample_tensor(sample):
    # Make a deep copy of the sample.
    sc = copy.deepcopy(sample)

    # Change the data type from PyTorch tensor to NumPy array.
    sc["image0"] = sc["image0"].numpy().transpose((1, 2, 0)).astype(np.uint8)
    sc["image1"] = sc["image1"].numpy().transpose((1, 2, 0)).astype(np.uint8)

    sc["disparity0"] = sc["disparity0"].numpy()[0, :, :]
    sc["disparity1"] = sc["disparity1"].numpy()[0, :, :]

    sc["disparity0"] = sc["disparity0"] / sc["disparity0"].max()
    sc["disparity1"] = sc["disparity1"] / sc["disparity1"].max()

    print("disparity0 range = (%f, %f)." % ( sc["disparity0"].min(), sc["disparity0"].max() ))
    print("disparity1 range = (%f, %f)." % ( sc["disparity1"].min(), sc["disparity1"].max() ))

    show_sample(sc)

class Crop(object):
    def __init__(self, cList):
        """
        cList: A four-element list. [y, h]
        """
        self.cList = cList

        self.hStarting = self.cList[0]
        self.hEnding   = self.hStarting + self.cList[1]

    def __call__(self, sample):
        # Retreive the images and the depths.
        image0     = sample["image0"]
        image1     = sample["image1"]
        disparity0 = sample["disparity0"]
        disparity1 = sample["disparity1"]

        return {
            "image0": image0[self.hStarting:self.hEnding, :],\
            "image1": image1[self.hStarting:self.hEnding, :],\
            "disparity0": disparity0[self.hStarting:self.hEnding, :],\
            "disparity1": disparity1[self.hStarting:self.hEnding, :],\
            }

class DownsampleCrop(object):
    def __init__(self, newSize):
        self.newSize = newSize

    def __call__(self, sample):
        # Retreive the images and the depths.
        image0     = sample["image0"]
        image1     = sample["image1"]
        disparity0 = sample["disparity0"]
        disparity1 = sample["disparity1"]

        sizeOri = disparity0.shape

        # Downsample the iamges.
        # Evil design of OpenCV!!!, dszie argument of resize() must be (width, height)
        image0     = cv2.resize( image0,     (self.newSize[1], self.newSize[0]), interpolation = cv2.INTER_NEAREST )
        image1     = cv2.resize( image1,     (self.newSize[1], self.newSize[0]), interpolation = cv2.INTER_NEAREST )
        disparity0 = cv2.resize( disparity0, (self.newSize[1], self.newSize[0]), interpolation = cv2.INTER_NEAREST )
        disparity1 = cv2.resize( disparity1, (self.newSize[1], self.newSize[0]), interpolation = cv2.INTER_NEAREST )

        disparity0 = disparity0 * ( self.newSize[1] / sizeOri[1] )
        disparity1 = disparity1 * ( self.newSize[1] / sizeOri[1] )

        return { "image0": image0, "image1": image1, "disparity0": disparity0, "disparity1": disparity1 }

class Downsample(object):
    def __init__(self, newSize):
        assert isinstance( newSize, (int, tuple) )

        self.newSize = newSize

        if isinstance( self.newSize, int ):
            if ( self.newSize <= 0 ):
                raise Exception("The downsample size must be a positive integer.")
            
            self.ratio = [ 1.0 / newSize, 1.0 / newSize ]
        else:
            if ( self.newSize[0] <= 0 or self.newSize[1] <= 0 ):
                raise Exception("The downsample size must be two positive integers if using tuple.")
            
            if ( not isinstance( self.newSize[0], int ) or not isinstance( self.newSize[1], int ) ):
                raise Exception("The downsample size must be two integers.")
            
            self.ratio = [ 1.0 / newSize[0], 1.0 / newSize[1] ]

    def __call__(self, sample):
        # Retreive the images and the depths.
        image0     = sample["image0"]
        image1     = sample["image1"]
        disparity0 = sample["disparity0"]
        disparity1 = sample["disparity1"]

        # Downsample the iamges.
        image0     = cv2.resize( image0,     (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        image1     = cv2.resize( image1,     (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        disparity0 = cv2.resize( disparity0, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        disparity1 = cv2.resize( disparity1, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )

        disparity0 = disparity0 * self.ratio[0]
        disparity1 = disparity1 * self.ratio[0]

        return { "image0": image0, "image1": image1, "disparity0": disparity0, "disparity1": disparity1 }

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # Retreive the images and the depths.
        image0 = torch.from_numpy( sample["image0"].astype(np.float32).transpose((2, 0, 1)) )
        image1 = torch.from_numpy( sample["image1"].astype(np.float32).transpose((2, 0, 1)) )

        disparityShape = sample["disparity0"].shape

        disparity0 = torch.from_numpy( sample["disparity0"].astype(np.float32).reshape((1, disparityShape[0], disparityShape[1])) )
        disparity1 = torch.from_numpy( sample["disparity1"].astype(np.float32).reshape((1, disparityShape[0], disparityShape[1])) )

        return { "image0": image0, "image1": image1, "disparity0": disparity0, "disparity1": disparity1 }

class Normalize(object):
    def __init__(self, mean, std):
        """
        mean: An array like three-element object.
        std:  An array like three-element object.
        """

        self.mean = mean
        self.std  = std

    def __call__(self, sample, reverse = False):
        """
        NOTE: This function requires that the images in sample MUST be
        Tensor, and the dimensions are defined as (channel, height, width).

        NOTE: The normalization of the images are NOT done in-place for 
        debugging reason.
        """

        sampleCopied = copy.deepcopy( sample )

        image0, image1 = sampleCopied["image0"], sampleCopied["image1"]

        if ( False == reverse ):
            for t, m, s in zip(image0, self.mean, self.std):
                t.sub_(m).div_(s)

            for t, m, s in zip(image1, self.mean, self.std):
                t.sub_(m).div_(s)
        else:
            for t, m, s in zip(image0, self.mean, self.std):
                t.mul_(s).add_(m)

            for t, m, s in zip(image1, self.mean, self.std):
                t.mul_(s).add_(m)

        return sampleCopied

class MiddleburyDataset(Dataset):
    def __init__(self, rootDir, DSFiles = "DSFiles.json",\
        transform = None,\
        forcedStatisticsCalculation = False):
        self.rootDir   = rootDir
        self.DSFiles   = DSFiles
        self.transform = transform

        # Find all DSFiles.
        self.DSFilesList = find_filenames_recursively( self.rootDir, self.DSFiles )
        self.nDSFiles = len( self.DSFilesList )

        if ( 0 == self.nDSFiles ):
            raise Exception("Length of DSFileList is zero.")

        # Build a lookup table for the names.
        self.nameDict = {}
        self.make_name_dict()

        # Statistics. Note that if using OpenCV, the order of the color channels may be different.
        self.mean = [0, 0, 0] # RGB values.
        self.std  = [1, 1, 1] # Standard deviation values for the RGB.

        self.statFile = self.rootDir + "/StatisticsRGB.json"

        if ( True == forcedStatisticsCalculation ):
            self.get_statistics()
        elif ( not os.path.isfile( self.statFile ) ):
            self.get_statistics()
        else:
            # Read results from file.
            fp = open(self.statFile, "r")
            s = json.load(fp)
            self.mean = s["mean"]
            self.std  = s["std"]
            fp.close()
            print("Statistics loaded from %s." % (self.statFile))

    def make_name_dict(self):
        for i in range( len(self) ):
            dsf = self.DSFilesList[i]

            # Read the JSON file.
            fp = open( dsf, "r" )
            DSFiles = json.load(fp)
            fp.close()

            # Put the name element into self.nameDict.
            self.nameDict[ DSFiles["name"] ] = i

    def get_statistics(self):
        print("AirsimStereoDataset is evaluating the statistics. This may take some time.")

        # Clear the statistics.
        self.mean = [ 0, 0, 0 ]
        self.std  = [ 1, 1, 1 ]

        print("Calculating the mean value.")

        N0 = 0.0 # The number of pixels of the previous images.
        N1 = 0.0 # The number of pixels of the current images.

        # Loop over all the data.
        for i in range( len(self) ):
            sample = self[i]

            # Image0.
            image0 = sample["image0"]

            N1 = image0.shape[0] * image0.shape[1]

            for j in range( image0.shape[2] ):
                # Mean value of iamge0.
                m = np.mean( image0[ :, :, j] )
                # Calculate new mean value.
                self.mean[j] = N0 / ( N0 + N1 ) * self.mean[j] + N1 / ( N0 + N1 ) * m

            # Image1.
            image1 = sample["image1"]

            N1 = image1.shape[0] * image1.shape[1]

            for j in range( image1.shape[2] ):
                # Mean value of iamge1.
                m = np.mean( image1[ :, :, j] )
                # Calculate new mean value.
                self.mean[j] = N0 / ( N0 + N1 ) * self.mean[j] + N1 / ( N0 + N1 ) * m
        
            N0 += N1

        print("Mean values: {}.".format(self.mean))

        N0 = 0.0
        N1 = 0.0

        # Loop over all the data.
        for i in range( len(self) ):
            sample = self[i]

            # Image0.
            image0 = sample["image0"]

            # Total number of samples minus 1.
            N1 = image0.shape[0] * image0.shape[1] - 1

            for j in range( image0.shape[2] ):
                # Sample standard deviation of iamge0, squared.
                s = np.sum( np.power( image0[ :, :, j] - self.mean[j], 2 ) ) / N1
                # Calculate new mean value.
                self.std[j] = N0 / ( N0 + N1 ) * self.std[j] + N1 / ( N0 + N1 ) * s

            # Image1.
            image1 = sample["image1"]

            # Total number of samples minus 1.
            N1 = image1.shape[0] * image1.shape[1] - 1

            for j in range( image1.shape[2] ):
                # Sample standard deviation of iamge0, squared.
                s = np.sum( np.power( image1[ :, :, j] - self.mean[j], 2 ) ) / N1
                # Calculate new mean value.
                self.std[j] = N0 / ( N0 + N1 ) * self.std[j] + N1 / ( N0 + N1 ) * s
        
            N0 += N1

            # Square root.
            self.std[0] = math.sqrt( self.std[0] )
            self.std[1] = math.sqrt( self.std[1] )
            self.std[2] = math.sqrt( self.std[2] )

        print("Std values: {}.".format(self.std))

        # Save the result to a json file
        d = { "mean": self.mean, "std": self.std, "order": "bgr" }

        fp = open( self.statFile, "w" )
        json.dump(d, fp)
        fp.close()
        print( "Statistics saved to %s." % (self.statFile) )

    def __len__(self):
        return (int)( self.nDSFiles )

    def __getitem__(self, idx):
        """
        A dictionary is returned. The dictionary contains the 
        """
        
        # Load the DSFiles.json.
        fp = open( self.DSFilesList[idx], "r" )
        DSFiles = json.load(fp)
        fp.close()

        # Split the path and filename.
        p, _ = os.path.split( self.DSFilesList[idx] )

        # Load the images.
        img0 = cv2.imread( p + "/" + DSFiles["image0"] )
        img1 = cv2.imread( p + "/" + DSFiles["image1"] )

        # Load the disparity maps.
        dsp0 = cv2.imread( p + "/" + DSFiles["disparity0"], cv2.IMREAD_GRAYSCALE ) * DSFiles["disparityFactor"]
        dsp1 = cv2.imread( p + "/" + DSFiles["disparity1"], cv2.IMREAD_GRAYSCALE ) * DSFiles["disparityFactor"]

        sample = { "name": DSFiles["name"], "image0": img0, "image1": img1, "disparity0": dsp0, "disparity1": dsp1 }

        if ( self.transform is not None ):
            sample = self.transform( sample )

        return sample

    def get_sample_by_name(self, name):
        # Test if the name is ins self.nameDict.

        if ( False == (name in self.nameDict) ):
            raise Exception("%s does not exits in the name dict." % (name))

        return self[ self.nameDict[name] ]

if __name__ == "__main__":
    # Test the data loader object.
    mbd = MiddleburyDataset( "/home/yyhu/expansion/OriginalData/MiddleburyDataSets/stereo/2003", "DSFiles.json" )

    # Show the name dict.
    print(mbd.nameDict)

    sample0 = mbd.get_sample_by_name("conesF")

    # Get a sample.
    sample = mbd[1]

    # Create a Downsample object.
    ds = Downsample((2, 2))
    dsSample = ds(sample)

    # Create a ToTensor object.
    tt = ToTensor()
    ttSample = tt(dsSample)

    # Create a Normalize object.
    nm = Normalize(mbd.mean, mbd.std)
    nmSample = nm(ttSample)

    # Test transforms.
    cm = transforms.Compose( [ds, tt, nm] ) 
    cmSample = cm( sample )

    # Create a data loader with transform.
    mbdTs = MiddleburyDataset( "/home/yyhu/expansion/OriginalData/MiddleburyDataSets/stereo/2003", transform = cm )
    sampleTs = mbdTs[1]

    # Test with data loader.
    dataLoader = DataLoader( mbdTs, batch_size = 1, shuffle = False, num_workers = 2, drop_last = False )

    # Test the iterator-like functionality of dataLoader.
    dlIter = iter( dataLoader )
    
    while (True):
        try:
            bs = dlIter.next()
        except StopIteration as exp:
            print("Iteration stops.")
            break

    for idxBatch, batchedSample in enumerate( dataLoader ):
        print( "idx = {}, size = {}.".format(idxBatch, batchedSample["image0"].size()) )
