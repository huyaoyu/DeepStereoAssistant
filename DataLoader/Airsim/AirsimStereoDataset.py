
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

from ..DatasetUtils import find_filenames

def show_sample(sample):
    # Create a matplotlib figure.
    fig = plt.figure()

    axImgL = plt.subplot(2, 2, 1)
    plt.tight_layout()
    axImgL.set_title("Left")
    axImgL.axis("off")
    plt.imshow( sample["image0"] )

    axImgR = plt.subplot(2, 2, 2)
    plt.tight_layout()
    axImgR.set_title("Right")
    axImgR.axis("off")
    plt.imshow( sample["image1"] )

    axDispL = plt.subplot(2, 2, 3)
    plt.tight_layout()
    axDispL.set_title("DispLeft")
    axDispL.axis("off")
    plt.imshow( sample["disparity0"] )

    axDispR = plt.subplot(2, 2, 4)
    plt.tight_layout()
    axDispR.set_title("DispRight")
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
        depth0     = sample["depth0"]
        depth1     = sample["depth1"]
        disparity0 = sample["disparity0"]
        disparity1 = sample["disparity1"]

        # Downsample the iamges.
        image0     = cv2.resize( image0,     (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        image1     = cv2.resize( image1,     (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        depth0     = cv2.resize( depth0,     (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        depth1     = cv2.resize( depth1,     (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        disparity0 = cv2.resize( disparity0, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        disparity1 = cv2.resize( disparity1, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )

        disparity0 = disparity0 * self.ratio[0]
        disparity1 = disparity1 * self.ratio[0]

        return { "image0": image0, "image1": image1, "depth0": depth0, "depth1": depth1, "disparity0": disparity0, "disparity1": disparity1 }

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # Retreive the images and the depths.
        image0 = torch.from_numpy( sample["image0"].astype(np.float32).transpose((2, 0, 1)) )
        image1 = torch.from_numpy( sample["image1"].astype(np.float32).transpose((2, 0, 1)) )

        depthShape = sample["depth0"].shape

        depth0 = torch.from_numpy( sample["depth0"].astype(np.float32).reshape((1, depthShape[0], depthShape[1])) )
        depth1 = torch.from_numpy( sample["depth1"].astype(np.float32).reshape((1, depthShape[0], depthShape[1])) )

        disparityShape = sample["disparity0"].shape

        disparity0 = torch.from_numpy( sample["disparity0"].astype(np.float32).reshape((1, disparityShape[0], disparityShape[1])) )
        disparity1 = torch.from_numpy( sample["disparity1"].astype(np.float32).reshape((1, disparityShape[0], disparityShape[1])) )

        return { "image0": image0, "image1": image1, "depth0": depth0, "depth1": depth1, "disparity0": disparity0, "disparity1": disparity1 }

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

class AirsimStereoDataset(Dataset):
    def __init__(self, rootDir, imageDir, depthDir,\
        imageExt = "png", depthExt = "npy", imageSuffix = "_rgb", depthSuffix = "_depth", transform = None,\
        forcedStatisticsCalculation = False):
        self.rootDir     = rootDir
        self.imageDir    = imageDir
        self.depthDir    = depthDir
        self.imageExt    = imageExt    # File extension of image file.
        self.depthExt    = depthExt    # File extension of depth file.
        self.imageSuffix = imageSuffix
        self.depthSuffix = depthSuffix
        self.transform   = transform

        # Read the meta data.
        fp = open( self.rootDir + "/meta.json", "r" )
        self.meta = json.load(fp)
        fp.close()

        self.FB = self.meta["focalLength"] * self.meta["baseline"]

        # Find all the file names.
        self.imageFiles = find_filenames( \
            self.rootDir + "/" + self.imageDir,\
            "*." + self.imageExt )
        self.depthFiles = find_filenames( \
            self.rootDir + "/" + self.depthDir,\
            "*." + self.depthExt )

        # Check the number of files.
        self.nImageFiles = len( self.imageFiles )
        self.nDepthFiles = len( self.depthFiles )
        if ( self.nImageFiles != self.nDepthFiles ):
            raise Exception("The numbers of files found are not consistent with each other. nImageFiles = {}, nDepthFiles = {}".format(self.nImageFiles, self.nDepthFiles))
        
        if ( 0 == self.nImageFiles ):
            raise Exception("No files found.")

        if ( self.nImageFiles & 0x01 == 0x01 ):
            raise Exception("There must be even number of files.")

        if ( False == self.check_filenames( \
            self.imageFiles, self.depthFiles, self.imageSuffix, self.depthSuffix, self.imageExt, self.depthExt ) ):
            raise Exception("File names are not consistent.")

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
    
    def check_filenames(self, fnList0, fnList1, suffix0, suffix1, ext0, ext1):
        """
        This function checks the filenames in fnList0 and fnList1 one by one. If all the
        filenames are consistent, then this function returns True. It returns False
        otherwise. When camparing the filenames, the suffix of each filename will be ignored.

        It is assumed that the lengths of fnList0 and fnList1 are the same.
        """

        nFiles   = len(fnList0)
        nSuffix0 = len(suffix0)
        nSuffix1 = len(suffix1)
        nExt0    = len(ext0)
        nExt1    = len(ext1)

        for i in range(nFiles):
            f0 = os.path.basename( fnList0[i] )[:-nSuffix0-nExt0-1]
            f1 = os.path.basename( fnList1[i] )[:-nSuffix1-nExt1-1]

            if ( f0 != f1 ):
                return False
        
        return True

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
        return (int)( self.nImageFiles / 2 )

    def __getitem__(self, idx):
        """
        A dictionary is returned. The dictionary contains the 
        """
        # The real indices.
        idx0 = idx * 2
        idx1 = idx0 + 1

        # Load the images.
        img0 = cv2.imread( self.imageFiles[idx0] )
        img1 = cv2.imread( self.imageFiles[idx1] )

        # Load the depth files.
        dpt0 = np.load( self.depthFiles[idx0] )
        dpt1 = np.load( self.depthFiles[idx1] )

        # Calculate the disparities.
        dsp0 = self.FB / dpt0 * self.meta["disparityFactor"]
        dsp1 = self.FB / dpt1 * self.meta["disparityFactor"]

        # Check the validities of the disparities.
        if ( np.sum( np.logical_not( np.isfinite(dsp0) ) ) > 0):
            raise Exception("idx = %d, non-finite dsp0 found" % (idx))
        
        if ( np.sum( np.logical_not( np.isfinite(dsp1) ) ) > 0):
            raise Exception("idx = %d, non-finite dsp1 found" % (idx))

        sample = { "image0": img0, "image1": img1, "depth0": dpt0, "depth1": dpt1, "disparity0": dsp0, "disparity1": dsp1 }

        if ( self.transform is not None ):
            sample = self.transform( sample )

        return sample

if __name__ == "__main__":
    # Test the data loader object.
    asd = AirsimStereoDataset( "../data/airsim_oldtown_stereo_01", "image", "depth_plan" )

    # Get a sample.
    sample = asd[10]

    # Create a Downsample object.
    ds = Downsample((2, 2))
    dsSample = ds(sample)

    # Create a ToTensor object.
    tt = ToTensor()
    ttSample = tt(dsSample)

    # Create a Normalize object.
    nm = Normalize(asd.mean, asd.std)
    nmSample = nm(ttSample)

    # Test transforms.
    cm = transforms.Compose( [ds, tt, nm] ) 
    cmSample = cm( sample )

    # Create a data loader with transform.
    asdTs = AirsimStereoDataset( "../data/airsim_oldtown_stereo_01", "image", "depth_plan", transform = cm )
    sampleTs = asdTs[10]

    # Test with data loader.
    dataLoader = DataLoader( asdTs, batch_size = 4, shuffle = False, num_workers = 4, drop_last = False )

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
