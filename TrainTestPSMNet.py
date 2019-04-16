from __future__ import print_function

import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from workflow import WorkFlow, TorchFlow

from TrainTestBase import TrainTestBase

from model import PyramidNet

import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("TrainTestCSN: Environment variable DISPLAY is not present in the system.")
    print("TrainTestCSN: Switch the backend of matplotlib to agg.")

class TTPSMNet(TrainTestBase):
    def __init__(self, workingDir, frame=None):
        super(TTPSMNet, self).__init__( workingDir, frame )

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

            self.model = self.frame.load_model( self.model, modelFn )

        self.frame.logger.info("PSMNet has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        self.optimizer = optim.Adam( self.model.parameters(), lr=self.learningRate )

    # Overload parent's function.
    def train(self, imgL, imgR, disp, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not train with infer mode.")

        self.model.train()
        imgL = Variable( torch.FloatTensor(imgL) )
        imgR = Variable( torch.FloatTensor(imgR) )
        disp = Variable( torch.FloatTensor(disp) )

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
            plt.imshow( img0 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = imgR[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img1 = img1 - img1.min()
            img1 = img1 / img1.max()
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
            plt.imshow( img0 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = imgR[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img1 = img1 - img1.min()
            img1 = img1 / img1.max()
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

    def infer(self, imgL, imgR):
        self.check_frame()

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

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
            self.frame.save_model( self.model, "PSMNet" )

class TTPSMNU(TTPSMNet):
    def __init__(self, workingDir, frame=None):
        super(TTPSMNU, self).__init__( workingDir, frame )

    # Overload parent's function.
    def init_model(self):
        if ( self.maxDisp <= 0 ):
            raise Exception("The maximum disparity must be positive.")

        # Neural net.
        if ( True == self.flagGrayscale ):
            self.model = PyramidNet.PSMNetWithUncertainty(1, 32, self.maxDisp)
        else:
            self.model = PyramidNet.PSMNetWithUncertainty(3, 32, self.maxDisp)

        # import ipdb; ipdb.set_trace()
        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                raise Exception("Model file (%s) does not exist." % ( modelFn ))

            self.model = self.frame.load_model( self.model, modelFn )

        self.frame.logger.info("PSMNet has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )

    # Overload parent's function.
    def train(self, imgL, imgR, disp, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not train with infer mode.")

        self.model.train()
        imgL = Variable( torch.FloatTensor(imgL) )
        imgR = Variable( torch.FloatTensor(imgR) )
        disp = Variable( torch.FloatTensor(disp) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp = disp.cuda()

        self.optimizer.zero_grad()

        out1, out2, out3, logSigSqu = self.model(imgL, imgR)

        out1 = torch.squeeze( out1, 1 )
        out2 = torch.squeeze( out2, 1 )
        out3 = torch.squeeze( out3, 1 )

        dispStartingIndex = disp.shape[1] - out1.shape[1]

        disp = disp[ :, dispStartingIndex:, :]

        mask = disp < self.maxDisp
        mask.detach_()

        # =================== New loss function. =============================
        expLogSigSqu = torch.exp(-logSigSqu)
        mExpLogSigSqu = expLogSigSqu[mask]

        loss = 0.5*F.smooth_l1_loss(mExpLogSigSqu * out1[mask], mExpLogSigSqu * disp[mask], reduction="mean") \
             + 0.7*F.smooth_l1_loss(mExpLogSigSqu * out2[mask], mExpLogSigSqu * disp[mask], reduction="mean") \
             +     F.smooth_l1_loss(mExpLogSigSqu * out3[mask], mExpLogSigSqu * disp[mask], reduction="mean")

        avgLogSigSqu = torch.mean( logSigSqu )
        self.frame.logger.info("avgLogSigSqu = %f." % (avgLogSigSqu.item()))
        loss = ( loss + avgLogSigSqu ) / 2.0

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

    # Overload parent's function.
    def test(self, imgL, imgR, disp, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not test in the infer mode.")

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        with torch.no_grad():
            output3, logSigSqu = self.model( imgL, imgR )

        output = torch.squeeze( output3.data.cpu(), 1 )

        dispStartingIndex = disp.shape[1] - output.shape[1]

        disp = disp[ :, dispStartingIndex:, :]

        mask = disp < self.maxDisp
        mask.detach_()

        if ( len( disp[mask] ) == 0 ):
            loss = 0
        else:
            expLogSigSqu = torch.exp(-logSigSqu)
            mExpLogSigSqu = expLogSigSqu[mask]

            loss = torch.mean( torch.abs( mExpLogSigSqu * output[mask] - mExpLogSigSqu * disp[mask] ) )
            loss = ( loss + torch.mean( logSigSqu ) ) / 2.0

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

    def infer(self, imgL, imgR):
        self.check_frame()

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        with torch.no_grad():
            output3, logSigSqu = self.model( imgL, imgR )

        output = torch.squeeze( output3.data.cpu(), 1 )

        self.countTest += 1

        # Draw and save results.
        identifier = "infer_%d" % (self.countTest - 1)
        self.draw_infer_results( identifier, output, imgL, imgR )
