
from __future__ import print_function

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

from workflow import WorkFlow, TorchFlow

from DataLoader.SceneFlow import Loader as DA
from model import PyramidNet

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

# Template for custom WorkFlow object.
class MyWF(TorchFlow.TorchFlow):
    def __init__(self, workingDir, prefix = "", suffix = ""):
        super(MyWF, self).__init__(workingDir, prefix, suffix)

        # === Custom member variables. ===

        # === Create the AccumulatedObjects. ===
        self.add_accumulated_value("lossTest", 10)

        # === Create a AccumulatedValuePlotter object for ploting. ===
        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "loss", self.AV, ["loss"], [True])\
        )

        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "lossTest", self.AV, ["lossTest"], [True])\
        )

        # === Custom member variables. ===
        self.countTrain = 0
        self.countTest  = 0

        self.imgTrainLoader = None
        self.imgTestLoader  = None

        self.model = None
        self.optimizer = None

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        # === Custom code. ===

        self.logger.info("Configure Torch.")

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        # Get all the sample images.
        imgTrainL, imgTrainR, dispTrain, imgTestL, imgTestR, dispTest \
            = list_files_sample("/home/yyhu/expansion/OriginalData/SceneFlow/Sampler/FlyingThings3D")

        # Dataloader.
        self.imgTrainLoader = torch.utils.data.DataLoader( \
            DA.myImageFolder( imgTrainL, imgTrainR, dispTrain, True ), \
            batch_size=2, shuffle=True, num_workers=2, drop_last=False )

        self.imgTestLoader = torch.utils.data.DataLoader( \
            DA.myImageFolder( imgTrainL, imgTrainR, dispTrain, False ), \
            batch_size=2, shuffle=False, num_workers=2, drop_last=False )

        # Neural net.
        self.model = PyramidNet.PSMNet(3, 32, 8)
        self.model.cuda()

        self.logger.info("PSMNet has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )

        self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )

        self.logger.info("Initialized.")
        self.post_initialize()

    # Overload the function train().
    def train(self, imgL, imgR, disp):
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

        loss = 0.5*F.smooth_l1_loss(out1[mask], disp[mask], size_average=True) \
             + 0.7*F.smooth_l1_loss(out2[mask], disp[mask], size_average=True) \
             +     F.smooth_l1_loss(out3[mask], disp[mask], size_average=True)

        loss.backward()

        self.optimizer.step()

        self.AV["loss"].push_back( loss.data[0] )

        self.countTrain += 1

        if ( self.countTrain % 10 == 0 ):
            self.write_accumulated_values()

        # Plot accumulated values.
        self.plot_accumulated_values()

        self.logger.info("Loop #%d %s" % (self.countTrain, self.get_log_str()))

    # Overload the function test().
    def test(self):
        super(MyWF, self).test()

        # === Custom code. ===
        # Test the existance of an AccumulatedValue object.
        if ( True == self.have_accumulated_value("lossTest") ):
            self.AV["lossTest"].push_back(0.01, self.countTest)
        else:
            self.logger.info("Could not find \"lossTest\"")

        self.logger.info("Tested.")

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        # === Custom code. ===
        self.logger.info("Finalized.")

if __name__ == "__main__":
    print("Hello TyrPSMNet.")

    print_delimeter(title = "Before WorkFlow initialization." )

    try:
        # Instantiate an object for MyWF.
        wf = MyWF("./Debug", prefix = "Debug_", suffix = "_debug")
        wf.verbose = True

        # Initialization.
        print_delimeter(title = "Initialize.")
        wf.initialize()

        # Training loop.
        print_delimeter(title = "Training loops.")

        for batchIdx, ( imgCropL, imgCropR, dispCrop ) in enumerate( wf.imgTrainLoader ):
            wf.train( imgCropL, imgCropR, dispCrop )

        # # Test and finalize.
        # print_delimeter(title = "Test and finalize.")

        # wf.test()
        wf.finalize()
    except WrokFlow.SigIntException as sie:
        print("SigInt revieved, perform finalize...")
        wf.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")