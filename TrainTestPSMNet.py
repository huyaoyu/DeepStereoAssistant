from __future__ import print_function

import copy
import cv2
import numpy as np
import os
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from workflow import WorkFlow, TorchFlow

from TrainTestBase import TrainTestBase

from model import PyramidNet
from PointCloud.PLYHelper import write_PLY
from TorchCUDAMem import TorchTraceMalloc

import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("TrainTestCSN: Environment variable DISPLAY is not present in the system.")
    print("TrainTestCSN: Switch the backend of matplotlib to agg.")

Q_FLIP = np.array( [ \
    [ 1.0,  0.0,  0.0, 0.0 ], \
    [ 0.0, -1.0,  0.0, 0.0 ], \
    [ 0.0,  0.0, -1.0, 0.0 ], \
    [ 0.0,  0.0,  0.0, 1.0 ] ], dtype=np.float32 )

class TTPSMNet(TrainTestBase):
    def __init__(self, workingDir, frame=None, modelName='PSMNet'):
        super(TTPSMNet, self).__init__( workingDir, frame, modelName )

    # def initialize(self):
    #     self.check_frame()
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_workflow(self):
        # === Create the AccumulatedObjects. ===
        self.frame.add_accumulated_value("lossTest", 10)

        self.frame.AV["loss"].avgWidth = 10
        
        # ======= AVP. ======
        # === Create a AccumulatedValuePlotter object for ploting. ===
        if ( True == self.flagUseIntPlotter ):
            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "loss", self.frame.AV, ["loss"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "lossTest", self.frame.AV, ["lossTest"], [True], semiLog=True) )
        else:
            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "loss", self.frame.AV, ["loss"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "lossTest", self.frame.AV, ["lossTest"], [True], semiLog=True) )

    # def init_workflow(self):
    #     raise Exception("Not implemented.")

    # def init_torch(self):
    #     raise Exception("Not implemented.")

    # def init_data(self):
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_model(self):
        if ( self.maxDisp <= 0 ):
            raise Exception("The maximum disparity must be positive.")

        # Neural net.
        if ( True == self.flagGrayscale ):
            self.model = PyramidNet.PSMNet(1, 32, self.maxDisp)
        else:
            self.model = PyramidNet.PSMNet(3, 32, self.maxDisp)

        # import ipdb; ipdb.set_trace()
        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                raise Exception("Model file (%s) does not exist." % ( modelFn ))

            self.frame.load_model( self.model, modelFn )

        if ( self.flagCPU ):
            self.model.set_cpu_mode()

        self.frame.logger.info("PSMNet has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     raise Exception("Not implemented.")

    def update_learning_rate(self):
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.learningRate

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        self.optimizer = optim.Adam( self.model.parameters(), lr=self.learningRate )

        # Check if we have to read the optimizer state from the filesystem.
        if ( "" != self.readOptimizerString ):
            optFn = "%s/models/%s" % ( self.frame.workingDir, self.readOptimizerString )

            if ( not os.path.isfile( optFn ) ):
                raise Exception("Optimizer file (%s) does not exist. " % ( optFn ))

            self.frame.load_optimizer(self.optimizer, optFn)
            # Update the learning rate.
            self.update_learning_rate()
            self.frame.logger.info("Optimizer state loaded for file %s. " % (optFn))

    # Overload parent's function.
    def train(self, imgL, imgR, disp, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not train with infer mode.")

        self.model.train()
        imgL = Variable( torch.FloatTensor(imgL) )
        imgR = Variable( torch.FloatTensor(imgR) )
        disp = Variable( torch.FloatTensor(disp) )

        if ( not self.flagCPU ):
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            disp = disp.cuda()

        self.optimizer.zero_grad()

        out1, out2, out3 = self.model(imgL, imgR)

        out1 = torch.squeeze( out1, 1 )
        out2 = torch.squeeze( out2, 1 )
        out3 = torch.squeeze( out3, 1 )

        dispStartingIndex = disp.shape[1] - out1.shape[1]

        disp = disp[ :, dispStartingIndex:, :]

        mask = disp < self.maxDisp
        mask.detach_()

        loss = 0.5*F.smooth_l1_loss(out1[mask], disp[mask], reduction="mean") \
             + 0.7*F.smooth_l1_loss(out2[mask], disp[mask], reduction="mean") \
             +     F.smooth_l1_loss(out3[mask], disp[mask], reduction="mean")

        # loss = F.smooth_l1_loss(out1[mask], disp[mask], reduction="mean")

        loss.backward()

        self.optimizer.step()

        self.frame.AV["loss"].push_back( loss.item() )

        self.countTrain += 1

        if ( self.countTrain % self.trainIntervalAccWrite == 0 ):
            self.frame.write_accumulated_values()

        # Plot accumulated values.
        if ( self.countTrain % self.trainIntervalAccPlot == 0 ):
            self.frame.plot_accumulated_values()

        # Auto-save.
        if ( 0 != self.autoSaveModelLoops ):
            if ( self.countTrain % self.autoSaveModelLoops == 0 ):
                modelName = "AutoSave_%08d" % ( self.countTrain )
                self.frame.logger.info("Auto-save the model.")
                self.frame.save_model( self.model, modelName )

        self.frame.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.frame.get_log_str()))

    def draw_test_results(self, identifier, predD, trueD, imgL, imgR):
        """
        Draw test results.

        predD: Dimension (B, H, W)
        trueD: Dimension (B, H, W)
        imgL: Dimension (B, C, H, W).
        imgR: Dimension (B, C, H, W).
        """

        batchSize = predD.size()[0]
        
        for i in range(batchSize):
            outDisp = predD[i, :, :].detach().cpu().numpy()
            gdtDisp = trueD[i, :, :].detach().cpu().numpy()

            gdtMin = gdtDisp.min()
            gdtMax = gdtDisp.max()

            # outDisp = outDisp - outDisp.min()
            outDisp = outDisp - gdtMin
            gdtDisp = gdtDisp - gdtMin

            # outDisp = outDisp / outDisp.max()
            outDisp = np.clip( outDisp / gdtMax, 0.0, 1.0 )
            gdtDisp = gdtDisp / gdtMax

            # Create a matplotlib figure.
            fig = plt.figure(figsize=(12.8, 9.6), dpi=300)

            ax = plt.subplot(2, 2, 1)
            plt.tight_layout()
            ax.set_title("Ref")
            ax.axis("off")
            img0 = imgL[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img0 = img0 - img0.min()
            img0 = img0 / img0.max()
            if ( 1 == img0.shape[2] ):
                img0 = img0[:, :, 0]
            plt.imshow( img0 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = imgR[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img1 = img1 - img1.min()
            img1 = img1 / img1.max()
            if ( 1 == img1.shape[2] ):
                img1 = img1[:, :, 0]
            plt.imshow( img1 )

            ax = plt.subplot(2, 2, 2)
            plt.tight_layout()
            ax.set_title("Ground truth")
            ax.axis("off")
            plt.imshow( gdtDisp )

            ax = plt.subplot(2, 2, 4)
            plt.tight_layout()
            ax.set_title("Prediction")
            ax.axis("off")
            plt.imshow( outDisp )

            figName = "%s_%02d" % (identifier, i)
            figName = self.frame.compose_file_name(figName, "png", subFolder=self.testResultSubfolder)
            plt.savefig(figName)

            plt.close(fig)

    def draw_infer_results(self, identifier, predD, imgL, imgR):
        """
        Draw test results.

        predD: Dimension (B, H, W)
        imgL: Dimension (B, C, H, W).
        imgR: Dimension (B, C, H, W).
        """

        batchSize = predD.size()[0]
        
        for i in range(batchSize):
            outDisp = predD[i, :, :].detach().cpu().numpy()

            outMin = outDisp.min()
            outMax = outDisp.max()

            # outDisp = outDisp - outDisp.min()
            outDisp = outDisp - outMin

            # outDisp = outDisp / outDisp.max()
            outDisp = np.clip( outDisp / outMax, 0.0, 1.0 )

            # Create a matplotlib figure.
            fig = plt.figure(figsize=(12.8, 9.6), dpi=300)

            ax = plt.subplot(2, 2, 1)
            plt.tight_layout()
            ax.set_title("Ref")
            ax.axis("off")
            img0 = imgL[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img0 = img0 - img0.min()
            img0 = img0 / img0.max()
            if ( 1 == img0.shape[2] ):
                img0 = img0[:, :, 0]
            plt.imshow( img0 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = imgR[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img1 = img1 - img1.min()
            img1 = img1 / img1.max()
            if ( 1 == img1.shape[2] ):
                img1 = img1[:, :, 0]
            plt.imshow( img1 )

            ax = plt.subplot(2, 2, 4)
            plt.tight_layout()
            ax.set_title("Prediction")
            ax.axis("off")
            plt.imshow( outDisp )

            figName = "%s_%02d" % (identifier, i)
            figName = self.frame.compose_file_name(figName, "png", subFolder=self.testResultSubfolder)
            plt.savefig(figName)

            plt.close(fig)

    # Overload parent's function.
    def test(self, imgL, imgR, disp, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not test in the infer mode.")

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        if ( not self.flagCPU ):
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        with torch.no_grad():
            output3 = self.model( imgL, imgR )

        output = torch.squeeze( output3.data.cpu(), 1 )

        dispStartingIndex = disp.shape[1] - output.shape[1]

        disp = disp[ :, dispStartingIndex:, :]

        mask = disp < self.maxDisp
        mask.detach_()

        if ( len( disp[mask] ) == 0 ):
            loss = 0
        else:
            loss = torch.mean( torch.abs( output[mask] - disp[mask] ) )

        self.countTest += 1

        if ( True == self.flagTest ):
            count = self.countTest
        else:
            count = self.countTrain

        # Draw and save results.
        identifier = "test_%d" % (count - 1)
        self.draw_test_results( identifier, output, disp, imgL, imgR )

        # Test the existance of an AccumulatedValue object.
        if ( True == self.frame.have_accumulated_value("lossTest") ):
            self.frame.AV["lossTest"].push_back(loss.item(), self.countTest)
        else:
            self.frame.logger.info("Could not find \"lossTest\"")

        self.frame.plot_accumulated_values()

        return loss.item()

    def infer(self, imgL, imgR, Q):
        self.check_frame()

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        if ( not self.flagCPU ):
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        with torch.no_grad():
            output3 = self.model( imgL, imgR )

        output = torch.squeeze( output3.data.cpu(), 1 )

        self.countTest += 1

        # Draw and save results.
        identifier = "infer_%d" % (self.countTest - 1)
        self.draw_infer_results( identifier, output, imgL, imgR )

    # Overload parent's function.
    def finalize(self):
        self.check_frame()

        # Save the model.
        if ( False == self.flagTest and False == self.flagInfer ):
            self.frame.save_model( self.model, self.modelName )
            self.frame.save_optimizer( self.optimizer, "%s_Opt" % (self.modelName) )

class TTPSMNU(TTPSMNet):
    def __init__(self, workingDir, frame=None, modelName='PSMNU'):
        super(TTPSMNU, self).__init__( workingDir, frame, modelName )

        self.flagInspect = False # Set True to perform inspection. NOTE: High filesystem memory consumption.
        self.inspector   = None

    # Overload parent's function.
    def init_workflow(self):
        # === Create the AccumulatedObjects. ===
        self.frame.add_accumulated_value("lossTest", 10)
        self.frame.add_accumulated_value("sigma2", 10)

        self.frame.AV["loss"].avgWidth = 10
        
        # ======= AVP. ======
        # === Create a AccumulatedValuePlotter object for ploting. ===
        if ( True == self.flagUseIntPlotter ):
            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "loss", self.frame.AV, ["loss"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "lossTest", self.frame.AV, ["lossTest"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "sigma2", self.frame.AV, ["sigma2"], [True], semiLog=True) )
        else:
            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "loss", self.frame.AV, ["loss"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "lossTest", self.frame.AV, ["lossTest"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "sigma2", self.frame.AV, ["sigma2"], [True], semiLog=True) )

    # Overload parent's function.
    def init_model(self):
        if ( self.maxDisp <= 0 ):
            raise Exception("The maximum disparity must be positive.")

        # Neural net.
        if ( False == self.flagInspect ):
            if ( True == self.flagGrayscale ):
                self.model = PyramidNet.PSMNetWithUncertainty(1, 32, self.maxDisp)
            else:
                self.model = PyramidNet.PSMNetWithUncertainty(3, 32, self.maxDisp)
            
            self.inspector = None
        else:
            if ( True == self.flagGrayscale ):
                self.model = PyramidNet.PSMNU_Inspect(1, 32, self.maxDisp)
            else:
                self.model = PyramidNet.PSMNU_Inspect(3, 32, self.maxDisp)

            self.inspector = PyramidNet.Inspector( self.frame.workingDir + "/Inspect" )

        # import ipdb; ipdb.set_trace()
        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                raise Exception("Model file (%s) does not exist." % ( modelFn ))

            self.frame.load_model( self.model, modelFn )

        if ( self.flagCPU ):
            self.model.set_cpu_mode()

        # Initialize the working directory for inspection.
        if ( True == self.flagInspect ):
            self.inspector.initialize_working_dir()

        self.frame.logger.info("PSMNet has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )

    # Overload parent's function.
    def train(self, imgL, imgR, disp, dispOri, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not train with infer mode.")

        # Increase counter.
        self.countTrain += 1

        self.model.train()
        imgL = Variable( torch.FloatTensor(imgL) )
        imgR = Variable( torch.FloatTensor(imgR) )
        disp = Variable( torch.FloatTensor(disp) )

        if ( not self.flagCPU ):
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            disp = disp.cuda()

        self.optimizer.zero_grad()

        if ( False == self.flagInspect ):
            out1, out2, out3, logSigSqu = self.model(imgL, imgR)
        else:
            prefix = "%s_Tr%d" % ( self.frame.prefix, self.countTrain )
            out1, out2, out3, logSigSqu = self.model(imgL, imgR, prefix, self.inspector)

        out1 = torch.squeeze( out1, 1 )
        out2 = torch.squeeze( out2, 1 )
        out3 = torch.squeeze( out3, 1 )

        # dispStartingIndex = disp.shape[1] - out1.shape[1]

        # disp = disp[ :, dispStartingIndex:, :]

        mask = (disp > 0) & (disp < self.maxDisp)
        mask.detach_()

        # =================== New loss function. =============================
        expLogSigSqu = torch.exp(-logSigSqu)
        mExpLogSigSqu = expLogSigSqu[mask]

        loss = 0.5*F.smooth_l1_loss(mExpLogSigSqu * out1[mask], mExpLogSigSqu * disp[mask], reduction="mean") \
             + 0.7*F.smooth_l1_loss(mExpLogSigSqu * out2[mask], mExpLogSigSqu * disp[mask], reduction="mean") \
             +     F.smooth_l1_loss(mExpLogSigSqu * out3[mask], mExpLogSigSqu * disp[mask], reduction="mean")

        avgLogSigSqu = torch.mean( logSigSqu )
        # self.frame.logger.info("avgLogSigSqu = %f." % (avgLogSigSqu.item()))
        loss = ( loss + avgLogSigSqu ) / 2.0

        # loss = F.smooth_l1_loss(out1[mask], disp[mask], reduction="mean")

        loss.backward()

        self.optimizer.step()

        self.frame.AV["loss"].push_back( loss.item() )
        self.frame.AV["sigma2"].push_back( torch.exp( avgLogSigSqu ).item() )

        if ( self.countTrain % self.trainIntervalAccWrite == 0 ):
            self.frame.write_accumulated_values()

        # Plot accumulated values.
        if ( self.countTrain % self.trainIntervalAccPlot == 0 ):
            self.frame.plot_accumulated_values()

        # Auto-save.
        if ( 0 != self.autoSaveModelLoops ):
            if ( self.countTrain % self.autoSaveModelLoops == 0 ):
                modelName = "AutoSave_%08d" % ( self.countTrain )
                self.frame.logger.info("Auto-save the model.")
                self.frame.save_model( self.model, modelName )

        self.frame.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.frame.get_log_str()))

    def draw_test_results(self, identifier, predD, trueD, imgL, imgR, logSigSqu, flagSaveDisp=False):
        """
        Draw test results.

        predD: Dimension (B, H, W)
        trueD: Dimension (B, H, W)
        imgL: Dimension (B, C, H, W).
        imgR: Dimension (B, C, H, W).
        logSigSqu: Dimension (B, H, W).
        """

        dispID = "%s_disp" % ( identifier )
        sigID  = "%s_sig" % ( identifier )

        batchSize = predD.size()[0]
        
        for i in range(batchSize):
            outDisp = predD[i, :, :].detach().cpu().numpy()
            gdtDisp = trueD[i, :, :].detach().cpu().numpy()

            if ( flagSaveDisp ):
                dispFn = "%s_%02d" % (dispID, i)
                dispFn = self.frame.compose_file_name(dispFn, "npy", subFolder=self.testResultSubfolder)
                np.save(dispFn, outDisp)

            gdtMin = gdtDisp.min()
            gdtMax = gdtDisp.max()

            # outDisp = outDisp - outDisp.min()
            outDisp = outDisp - gdtMin
            gdtDisp = gdtDisp - gdtMin

            # outDisp = outDisp / outDisp.max()
            outDisp = np.clip( outDisp / gdtMax, 0.0, 1.0 )
            gdtDisp = gdtDisp / gdtMax

            # Create a matplotlib figure.
            fig = plt.figure(figsize=(12.8, 9.6), dpi=300)

            ax = plt.subplot(2, 2, 1)
            plt.tight_layout()
            ax.set_title("Ref")
            ax.axis("off")
            img0 = imgL[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img0 = img0 - img0.min()
            img0 = img0 / img0.max()
            if ( 1 == img0.shape[2] ):
                img0 = img0[:, :, 0]
            else:
                # Assuming it is a 3-channel image.
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

            plt.imshow( img0 )

            # ax = plt.subplot(2, 2, 3)
            # plt.tight_layout()
            # ax.set_title("Tst")
            # ax.axis("off")
            # img1 = imgR[i, :, :, :].permute((1,2,0)).cpu().numpy()
            # img1 = img1 - img1.min()
            # img1 = img1 / img1.max()
            # plt.imshow( img1 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Sigma")
            ax.axis("off")

            # Calculate sigma.
            sigma = torch.squeeze( torch.sqrt( torch.exp( logSigSqu ) ), 0 ).cpu().numpy()

            if ( flagSaveDisp ):
                sigFn = "%s_%02d" % (sigID, i)
                sigFn = self.frame.compose_file_name(sigFn, "npy", subFolder=self.testResultSubfolder)
                np.save(sigFn, sigma)

            sigma = sigma - sigma.min()
            sigma = sigma / sigma.max()
            plt.imshow( sigma )

            ax = plt.subplot(2, 2, 2)
            plt.tight_layout()
            ax.set_title("Ground truth")
            ax.axis("off")
            plt.imshow( gdtDisp )

            ax = plt.subplot(2, 2, 4)
            plt.tight_layout()
            ax.set_title("Prediction")
            ax.axis("off")
            plt.imshow( outDisp )

            figName = "%s_%02d" % (identifier, i)
            figName = self.frame.compose_file_name(figName, "png", subFolder=self.testResultSubfolder)
            plt.savefig(figName)

            plt.close(fig)

    # Overload parent's function.
    def test(self, imgL, imgR, disp, dispOri, flagSaveDisp=False):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not test in the infer mode.")

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        if ( not self.flagCPU ):
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            disp = disp.cuda()
            dispOri = dispOri.cuda()

        with torch.no_grad():
            if ( False == self.flagInspect ):
                output3, logSigSqu = self.model( imgL, imgR )
            else:
                prefix = "%s_Te%d" % ( self.frame.prefix, self.countTest )
                output3, logSigSqu = self.model( imgL, imgR, prefix, self.inspector )

        output = torch.unsqueeze(output3, 1)
        logSigSqu = torch.unsqueeze(logSigSqu, 1)

        # print("disp.size() = \n{}".format(disp.size()))
        # print("dispOri.size() = \n{}".format(dispOri.size()))
        # print("output.size() = \n{}".format(output.size()))
        # print("logSigSqu.size() = \n{}".format(logSigSqu.size()))
        # raise Exception("Test.")

        # Resize to the original size.
        hOri = dispOri.size()[1]
        wOri = dispOri.size()[2]

        wTest = output.size()[3]

        output = F.interpolate(output, (hOri, wOri), mode="bilinear", align_corners=False ) * (1.0*wOri/wTest)
        logSigSqu = F.interpolate(logSigSqu, (hOri, wOri), mode="bilinear", align_corners=False ) * (1.0*wOri/wTest)**2

        output = torch.squeeze( output, 1 )
        logSigSqu = torch.squeeze( logSigSqu, 1)

        logSigSqu.clamp_(-10, 10)
        # logSigSqu = logSigSqu.data.cpu()

        # self.frame.logger.info("disp.shape = \n{}".format(disp.shape))
        # self.frame.logger.info("output.shape = \n{}".format(output.shape))

        # dispStartingIndex = disp.shape[1] - output.shape[1]

        # disp = disp[ :, dispStartingIndex:, :]

        if ( True == self.flagTest ):
            count = self.countTest
        else:
            count = self.countTrain

        mask = ( dispOri > 0 ) & ( dispOri < self.maxDisp )
        mask.detach_()

        flagGoodLoss = 1

        if ( len( dispOri[mask] ) == 0 ):
            loss = torch.zeros((1)) # Dummy value.
            self.frame.logger.info("Test index %d. mask length = 0. " % (count))
            flagGoodLoss = 0
        else:
            # import ipdb; ipdb.set_trace()
            expLogSigSqu = torch.exp(-logSigSqu)
            mExpLogSigSqu = expLogSigSqu[mask]

            loss = torch.mean( torch.abs( mExpLogSigSqu * output[mask] - mExpLogSigSqu * dispOri[mask] ) )
            loss = ( loss + torch.mean( logSigSqu ) ) / 2.0

        # Draw and save results.
        identifier = "test_%04d" % (count)
        self.draw_test_results( identifier, output, dispOri, imgL, imgR, logSigSqu, flagSaveDisp )

        # Test the existance of an AccumulatedValue object.
        if ( True == self.frame.have_accumulated_value("lossTest") ):
            self.frame.AV["lossTest"].push_back(loss.item(), self.countTest)
        else:
            self.frame.logger.info("Could not find \"lossTest\"")

        self.frame.plot_accumulated_values()

        # Increase the counter.
        self.countTest += 1

        return loss.item(), flagGoodLoss

    def draw_infer_results(self, identifier, predD, imgL, imgR, logSigSqu, Q=None, flagSaveDisp=False):
        """
        Draw test results.

        predD: Dimension (B, H, W)
        imgL: Dimension (B, C, H, W).
        imgR: Dimension (B, C, H, W).
        """

        imgID  = "%s_img" % ( identifier )
        dispID = "%s_disp" % ( identifier  )
        sigID  = "%s_sig" % ( identifier )
        plyID  = "%s_ply" % ( identifier )

        batchSize = predD.size()[0]
        
        for i in range(batchSize):
            outDisp = predD[i, :, :].detach().cpu().numpy()

            if ( flagSaveDisp ):
                dispFn = "%s_%02d" % (dispID, i)
                dispFn = self.frame.compose_file_name(dispFn, "npy", subFolder=self.testResultSubfolder)
                np.save(dispFn, outDisp)

            dispOri = copy.deepcopy( outDisp )

            outMin = outDisp.min()
            outMax = outDisp.max()

            # outDisp = outDisp - outDisp.min()
            outDisp = outDisp - outMin

            # outDisp = outDisp / outDisp.max()
            outDisp = np.clip( outDisp / outMax, 0.0, 1.0 )

            # Create a matplotlib figure.
            fig = plt.figure(figsize=(12.8, 9.6), dpi=300)

            # Input reference image.
            ax = plt.subplot(2, 2, 1)
            plt.tight_layout()
            ax.set_title("Ref")
            ax.axis("off")
            imgOri = imgL[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img0 = copy.deepcopy( imgOri )
            img0 = img0 - img0.min()
            img0 = img0 / img0.max()
            if ( 1 == img0.shape[2] ):
                img0 = img0[:, :, 0]
            else:
                # Assuming it is a 3-channel image.
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

            plt.imshow( img0 )

            # Input test image.
            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = imgR[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img1 = img1 - img1.min()
            img1 = img1 / img1.max()
            if ( 1 == img1.shape[2] ):
                img1 = img1[:, :, 0]
            else:
                # Assuming it is a 3-channel image.
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            plt.imshow( img1 )

            ax = plt.subplot(2, 2, 4)
            plt.tight_layout()
            ax.set_title("Sigma")
            ax.axis("off")

            # Calculate sigma.
            sigma = torch.sqrt( torch.exp( logSigSqu[i, :, :] ) ).cpu().numpy()

            if ( flagSaveDisp ):
                sigFn = "%s_%02d" % (sigID, i)
                sigFn = self.frame.compose_file_name(sigFn, "npy", subFolder=self.testResultSubfolder)
                np.save(sigFn, sigma)
            
            sigma = sigma - sigma.min()
            sigma = sigma / sigma.max()
            plt.imshow( sigma )

            # Output disparity.
            ax = plt.subplot(2, 2, 2)
            plt.tight_layout()
            ax.set_title("Prediction")
            ax.axis("off")
            plt.imshow( outDisp )

            figName = "%s_%02d" % (imgID, i)
            figName = self.frame.compose_file_name(figName, "png", subFolder=self.testResultSubfolder)
            plt.savefig(figName)

            plt.close(fig)

            if ( Q is not None ):
                # Save the PLY file.
                q = Q[i, :, :].numpy()

                print("B{}, q=\n{}".format(i, q))

                plyFn = "%s_%02d" % ( plyID, i )
                plyFn = self.frame.compose_file_name( plyFn, "ply", subFolder=self.testResultSubfolder )
                
                write_PLY( plyFn, dispOri, q, color=img0 * 255)

    def infer(self, imgL, imgR, Q, imgLOri, flagSaveDisp=False, falgSaveCloud=False):
        self.check_frame()

        # Increase the counter.
        self.countTest += 1

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        if ( not self.flagCPU ):
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            imgLOri = imgLOri.cuda()

        with TorchTraceMalloc() as ttm:
            startT = time.time()
            with torch.no_grad():
                if ( False == self.flagInspect ):
                    output3, logSigSqu = self.model( imgL, imgR )
                else:
                    prefix = "%s_In%d" % ( self.frame.prefix, self.countTest - 1 )
                    output3, logSigSqu = self.model( imgL, imgR, prefix, self.inspector )

            endT = time.time()
            memSnap = np.array(ttm.snap_shot(), dtype=np.int)

        # Calculate elapsed time.
        et = endT - startT

        # Save the elapsed time.
        timeFn = "infer_time_%04d" % (self.countTest - 1)
        timeFn = self.frame.compose_file_name(timeFn, "txt", subFolder=self.testResultSubfolder)
        np.savetxt(timeFn, [et], fmt="%f")

        self.frame.logger.info("infer() time %f. " % ( et ))

        # Save the memory usage.
        memFn = "infer_mem_%04d" % (self.countTest - 1)
        memFn = self.frame.compose_file_name(memFn, 'txt', subFolder=self.testResultSubfolder)
        np.savetxt(memFn, memSnap)

        # import ipdb; ipdb.set_trace()
        output = torch.unsqueeze(output3, 1)
        logSigSqu = torch.unsqueeze(logSigSqu, 1)

        # Resize to the original size.
        # imgLOri is in B, H, W, C order. 20201112.
        hOri = imgLOri.size()[1]
        wOri = imgLOri.size()[2]

        wTest = output.size()[3]

        output = F.interpolate(output, (hOri, wOri), mode="bilinear", align_corners=False ) * (1.0*wOri/wTest)
        logSigSqu = F.interpolate(logSigSqu, (hOri, wOri), mode="bilinear", align_corners=False ) * (1.0*wOri/wTest)**2

        output = torch.squeeze( output, 1 )
        logSigSqu = torch.squeeze(logSigSqu, 1)

        if ( falgSaveCloud ):
            # Create helper tensor for flipping Q.
            QF = torch.from_numpy( Q_FLIP )

            for i in range( Q.shape[0] ):
                Q[i, :, :] = QF.mm( Q[i, :, :] )
        else:
            Q = None

        # Draw and save results.
        identifier = "infer_%04d" % (self.countTest - 1)
        self.draw_infer_results( identifier, output, imgL, imgR, logSigSqu, Q, flagSaveDisp )
        
