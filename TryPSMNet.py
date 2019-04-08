
from __future__ import print_function

import argparse
import glob
import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from workflow import WorkFlow, TorchFlow

import ArgumentParser
from DataLoader.SceneFlow import Loader as DA
from DataLoader import PreProcess
from model import PyramidNet

RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL = 100

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range( int(n/2) )]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

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

# Template for custom WorkFlow object.
class MyWF(TorchFlow.TorchFlow):
    def __init__(self, workingDir, prefix = "", suffix = "", disableStreamLogger=False):
        super(MyWF, self).__init__(workingDir, prefix, suffix, disableStreamLogger)

        # === Custom member variables. ===

        self.flagGrayscale = False

        # === Create the AccumulatedObjects. ===
        self.add_accumulated_value("lossTest", 10)

        self.AV["loss"].avgWidth = 10

        # NN.
        self.countTrain = 0
        self.countTest  = 0

        self.trainIntervalAccWrite = 10    # The interval to write the accumulated values.
        self.trainIntervalAccPlot  = 1     # The interval to plot the accumulate values.
        self.flagUseIntPlotter     = False # The flag of intermittent plotter.

        self.imgTrainLoader = None
        self.imgTestLoader  = None
        self.datasetRootDir = "./"
        self.dataEntries    = 0 # 0 for using all the data.
        self.datasetTrain   = None # Should be an object of torch.utils.data.Dataset.
        self.datasetTest    = None # Should be an object of torch.utils.data.Dataset.
        self.dlBatchSize    = 2
        self.dlShuffle      = True
        self.dlNumWorkers   = 2
        self.dlDropLast     = False

        self.model     = None
        self.multiGPUs = False

        self.readModelString    = ""
        self.autoSaveModelLoops = 0 # The number of loops to perform an auto-saving of the model. 0 for disable.

        self.optimizer = None

    def enable_multi_GPUs(self):
        self.multiGPUs = True

        self.logger.info("Enable multi-GPUs.")

    def set_dataset_root_dir(self, d, nEntries=0):
        if ( False == os.path.isdir(d) ):
            Exception("Dataset directory (%s) not exists." % (d))
        
        self.datasetRootDir = d
        self.dataEntries    = nEntries

        self.logger.info("Data root directory is %s." % ( self.datasetRootDir ))
        if ( 0 != nEntries ):
            self.logger.warning("Only %d entries of the training dataset will be used." % ( nEntries ))

    def set_data_loader_params(self, batchSize=2, shuffle=True, numWorkers=2, dropLast=False):
        self.dlBatchSize  = batchSize
        self.dlShuffle    = shuffle
        self.dlNumWorkers = numWorkers
        self.dlDropLast   = dropLast

    def set_read_model(self, readModelString):
        self.readModelString = readModelString

        if ( "" != self.readModelString ):
            self.logger.info("Read model from %s." % ( self.readModelString ))

    def enable_auto_save(self, loops):
        self.autoSaveModelLoops = loops

        if ( 0 != self.autoSaveModelLoops ):
            self.logger.info("Auto save enabled with loops = %d." % (self.autoSaveModelLoops))

    def set_training_acc_params(self, intervalWrite, intervalPlot, flagInt=False):
        self.trainIntervalAccWrite = intervalWrite
        self.trainIntervalAccPlot  = intervalPlot
        self.flagUseIntPlotter     = flagInt

        if ( True == self.flagUseIntPlotter ):
            if ( self.trainIntervalAccPlot <= RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ):
                self.logger.warning("When using the intermittent plotter. It is recommended that the plotting interval (%s) is higher than %d." % \
                    ( self.trainIntervalAccPlot, RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ) )

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        # === Custom code. ===

        self.logger.info("Configure Torch.")

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        # Get all the sample images.
        # imgTrainL, imgTrainR, dispTrain, imgTestL, imgTestR, dispTest \
        #     = list_files_sample("/home/yyhu/expansion/OriginalData/SceneFlow/Sampler/FlyingThings3D")
        # imgTrainL, imgTrainR, dispTrain, imgTestL, imgTestR, dispTest \
        #     = list_files_sample("/media/yaoyu/DiskE/SceneFlow/Sampler/FlyingThings3D")

        imgTrainL, imgTrainR, dispTrain, imgTestL, imgTestR, dispTest \
            = list_files_sceneflow_FlyingThings( self.datasetRootDir )

        if ( 0 != self.dataEntries ):
            imgTrainL = imgTrainL[0:self.dataEntries]
            imgTrainR = imgTrainR[0:self.dataEntries]
            dispTrain = dispTrain[0:self.dataEntries]
            imgTestL  = imgTestL[0:self.dataEntries]
            imgTestR  = imgTestR[0:self.dataEntries]
            dispTest  = dispTest[0:self.dataEntries]

        # Dataloader.
        if ( True == self.flagGrayscale ):
            preprocessor = transforms.Compose( [ \
                transforms.ToTensor(), \
                PreProcess.Grayscale(), \
                PreProcess.SingleChannel() ] )
        else:
            preprocessor = PreProcess.get_transform(augment=False)

        self.datasetTrain = DA.myImageFolder( imgTrainL, imgTrainR, dispTrain, True, preprocessor=preprocessor )
        self.datasetTest  = DA.myImageFolder( imgTestL,  imgTestR,  dispTest, False, preprocessor=preprocessor )

        self.imgTrainLoader = torch.utils.data.DataLoader( \
            self.datasetTrain, \
            batch_size=self.dlBatchSize, shuffle=self.dlShuffle, num_workers=self.dlNumWorkers, drop_last=self.dlDropLast )

        self.imgTestLoader = torch.utils.data.DataLoader( \
            self.datasetTest, \
            batch_size=self.dlBatchSize, shuffle=self.dlShuffle, num_workers=self.dlNumWorkers, drop_last=self.dlDropLast )

        # Neural net.
        if ( True == self.flagGrayscale ):
            self.model = PyramidNet.PSMNet(1, 32, 64)
        else:
            self.model = PyramidNet.PSMNet(3, 32, 64)

        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                Exception("Model file (%s) does not exist." % ( modelFn ))

            self.model = self.load_model( self.model, modelFn )

        if ( True == self.multiGPUs ):
            self.model = nn.DataParallel(self.model)

        self.model.cuda()

        self.logger.info("PSMNet has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )

        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        self.optimizer = optim.Adam( self.model.parameters(), lr=0.001 )

        # ======= AVP. ======
        # === Create a AccumulatedValuePlotter object for ploting. ===
        if ( True == self.flagUseIntPlotter ):
            self.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.workingDir + "/IntPlot", 
                    "loss", self.AV, ["loss"], [True], semiLog=True) )

            self.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.workingDir + "/IntPlot", 
                    "lossTest", self.AV, ["lossTest"], [True], semiLog=True) )
        else:
            self.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "loss", self.AV, ["loss"], [True], semiLog=True) )

            self.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "lossTest", self.AV, ["lossTest"], [True], semiLog=True) )

        self.logger.info("Initialized.")
        self.post_initialize()

    # Overload the function train().
    def train(self, imgL, imgR, disp, epochCount):
        super(MyWF, self).train()

        # === Custom code. ===

        self.model.train()
        imgL = Variable( torch.FloatTensor(imgL) )
        imgR = Variable( torch.FloatTensor(imgR) )
        disp = Variable( torch.FloatTensor(disp) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp = disp.cuda()

        mask = disp < 1000000
        mask.detach_()

        self.optimizer.zero_grad()

        out1, out2, out3 = self.model(imgL, imgR)

        out1 = torch.squeeze( out1, 1 )
        out2 = torch.squeeze( out2, 1 )
        out3 = torch.squeeze( out3, 1 )

        loss = 0.5*F.smooth_l1_loss(out1[mask], disp[mask], reduction="mean") \
             + 0.7*F.smooth_l1_loss(out2[mask], disp[mask], reduction="mean") \
             +     F.smooth_l1_loss(out3[mask], disp[mask], reduction="mean")

        # loss = F.smooth_l1_loss(out1[mask], disp[mask], reduction="mean")

        loss.backward()

        self.optimizer.step()

        self.AV["loss"].push_back( loss.item() )

        self.countTrain += 1

        if ( self.countTrain % self.trainIntervalAccWrite == 0 ):
            self.write_accumulated_values()

        # Plot accumulated values.
        if ( self.countTrain % self.trainIntervalAccPlot == 0 ):
            self.plot_accumulated_values()

        # Auto-save.
        if ( 0 != self.autoSaveModelLoops ):
            if ( self.countTrain % self.autoSaveModelLoops == 0 ):
                modelName = "AutoSave_%06d" % ( self.countTrain )
                self.logger.info("Auto-save the model.")
                self.save_model( self.model, modelName )

        self.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.get_log_str()))

    # Overload the function test().
    def test(self, imgL, imgR, disp, epochCount):
        super(MyWF, self).test()

        # === Custom code. ===

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        mask = disp < 192

        with torch.no_grad():
            output3 = self.model( imgL, imgR )

        output = torch.squeeze( output3.data.cpu(), 1 )[:, 4:, :]

        if ( len( disp[mask] ) == 0 ):
            loss = 0
        else:
            loss = torch.mean( torch.abs( output[mask] - disp[mask] ) )

        self.countTest += 1

        # Test the existance of an AccumulatedValue object.
        if ( True == self.have_accumulated_value("lossTest") ):
            self.AV["lossTest"].push_back(loss.item(), self.countTest)
        else:
            self.logger.info("Could not find \"lossTest\"")

        self.plot_accumulated_values()

        return loss.item()

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        # Save the model.
        self.save_model( self.model, "PSMNet" )

        # === Custom code. ===
        self.logger.info("Finalized.")

if __name__ == "__main__":
    print("Hello TyrPSMNet.")

    # Handle the arguments.
    args = ArgumentParser.args

    print_delimeter(title = "Before WorkFlow initialization." )

    try:
        # Instantiate an object for MyWF.
        wf = MyWF(args.working_dir, prefix=args.prefix, suffix=args.suffix, disableStreamLogger=False)
        wf.verbose = False

        if ( True == args.multi_gpus ):
            wf.enable_multi_GPUs()

        wf.flagGrayscale = args.grayscale

        # Set parameters.
        wf.set_data_loader_params( args.dl_batch_size, not args.dl_disable_shuffle, args.dl_num_workers, args.dl_drop_last )
        wf.set_dataset_root_dir( args.data_root_dir, args.data_entries )
        wf.set_read_model( args.read_model )
        wf.enable_auto_save( args.auto_save_model )
        wf.set_training_acc_params( args.train_interval_acc_write, args.train_interval_acc_plot, args.use_intermittent_plotter )

        # Initialization.
        print_delimeter(title = "Initialize.")
        wf.initialize()

        # Get the number of test data.
        nTests = len( wf.imgTestLoader )
        wf.logger.info("The size of the test dataset is %s." % ( nTests ))
        currentTestIdx = 0

        if ( False == args.test ):
            # Create the test data iterator.
            iterTestData = iter( wf.imgTestLoader )

            # Training loop.
            wf.logger.info("Begin training.")
            print_delimeter(title = "Training loops.")

            for i in range(args.train_epochs):
                for batchIdx, ( imgCropL, imgCropR, dispCrop ) in enumerate( wf.imgTrainLoader ):
                    # wf.logger.info( "imgCropL.shape = {}".format( imgCropL.shape ) )
                    # import ipdb; ipdb.set_trace()
                    wf.train( imgCropL, imgCropR, dispCrop, i )

                    # Check if we need a test.
                    if ( 0 != args.test_loops ):
                        if ( wf.countTrain % args.test_loops == 0 ):
                            # Get test data.
                            try:
                                testImgL, testImgR, testDisp = next( iterTestData )
                            except StopIteration:
                                iterTestData = iter(wf.imgTestLoader)
                                testImgL, testImgR, testDisp = next( iterTestData )

                            # Perform test.
                            wf.test( testImgL, testImgR, testDisp, i )
        else:
            wf.logger.info("Begin testing.")
            print_delimeter(title="Testing loops.")

            totalLoss = 0

            for batchIdx, ( imgL, imgR, disp ) in enumerate( wf.imgTestLoader ):
                loss = wf.test( imgL, imgR, disp, 0 )
                wf.logger.info("Test %d, loss = %f." % ( batchIdx, loss ))
                totalLoss += loss

            wf.logger.info("Average loss = %f." % ( totalLoss / nTests ))

        # wf.test()
        wf.finalize()
    except WorkFlow.SigIntException as sie:
        print("SigInt revieved, perform finalize...")
        wf.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")
