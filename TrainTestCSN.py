from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from workflow import WorkFlow, TorchFlow

from TrainTestBase import TrainTestBase

from model.CSN.ConvolutionalStereoNet import ConvolutionalStereoNet

import os
import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("TrainTestCSN: Environment variable DISPLAY is not present in the system.")
    print("TrainTestCSN: Switch the backend of matplotlib to agg.")

class TTCSN(TrainTestBase):
    def __init__(self, params, workingDir, frame=None):
        super(TTCSN, self).__init__( workingDir, frame )

        self.wd = workingDir
        self.params = params
        self.criterion = torch.nn.SmoothL1Loss()

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

        flagReadModel = ( "" != self.readModelString )

        # Neural net.
        if ( True == self.flagGrayscale ):
            raise Exception("Grayscale is not implemented for CSN yet.")
        else:
            self.model = ConvolutionalStereoNet( flagReadModel=flagReadModel, \
                preTrainedVGGPath=self.params["preTrainedVGG"] )

        # Check if we have to read the model from filesystem.
        if ( flagReadModel ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                raise Exception("Model file (%s) does not exist." % ( modelFn ))

            self.model = self.frame.load_model( self.model, modelFn )

        self.frame.logger.info("CSN has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        self.optimizer = optim.Adam( self.model.parameters(), lr=self.learningRate )

    def single_train(self, image0, image1, disparity0, md, cri, opt):
        """
        md:  The pytorch module.
        cir: The criterion.
        opt: The optimizer.
        """

        # Clear the gradients.
        opt.zero_grad()
        
        # Forward.
        output = md( image0, image1 )

        mask = disparity0 < self.maxDisp
        mask.detach_()

        output = output.squeeze( 1 )

        loss   = cri( output[mask], disparity0[mask] )

        # Handle the loss value.
        self.frame.AV["loss"].push_back( loss.item() )

        # Backward.
        loss.backward()
        opt.step()

    # Overload parent's function.
    def train(self, imgL, imgR, disp, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not train in the infer mode.")

        self.model.train()
        imgL = Variable( torch.FloatTensor(imgL) )
        imgR = Variable( torch.FloatTensor(imgR) )
        disp = Variable( torch.FloatTensor(disp) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp = disp.cuda()

        self.single_train( imgL, imgR, disp, self.model, self.criterion, self.optimizer )

        self.countTrain += 1

        if ( self.countTrain % self.trainIntervalAccWrite == 0 ):
            self.frame.write_accumulated_values()

        # Plot accumulated values.
        if ( self.countTrain % self.trainIntervalAccPlot == 0 ):
            self.frame.plot_accumulated_values()

        # Auto-save.
        if ( 0 != self.autoSaveModelLoops ):
            if ( self.countTrain % self.autoSaveModelLoops == 0 ):
                modelName = "AutoSave_%06d" % ( self.countTrain )
                self.frame.logger.info("Auto-save the model.")
                self.frame.save_model( self.model, modelName )

        self.frame.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.frame.get_log_str()))

    def single_test(self, identifier, image0, image1, disparity0, md, cri):
        """
        identifier: A string identifies this test.
        md:  The pytorch module.
        """
        
        # Forward.
        with torch.no_grad():
            output = md( image0, image1 )
            # output = output[:, :, 4:, :]
        
        outputTemp = torch.squeeze( output.data.cpu(), 1 )

        dispStartingIndex = disparity0.shape[1] - outputTemp.shape[1]

        disparity0 = disparity0[ :, dispStartingIndex:, :]

        mask = disparity0 < self.maxDisp
        mask.detach_()

        loss   = cri( outputTemp[mask], disparity0[mask] )

        # # Handle the loss value.
        # plotX = self.countTrain - 1
        # if ( plotX < 0 ):
        #     plotX = 0
        # self.frame.AV["lossTest"].push_back( loss.item(), plotX )

        # Save the test result.
        batchSize = output.size()[0]
        
        for i in range(batchSize):
            outDisp = output[i, 0, :, :].detach().cpu().numpy()
            gdtDisp = disparity0[i, :, :].detach().cpu().numpy()

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
            img0 = image0[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img0 = img0 - img0.min()
            img0 = img0 / img0.max()
            plt.imshow( img0 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = image1[i, :, :, :].permute((1,2,0)).cpu().numpy()
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

        return loss

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

        # mask = disp < 192

        self.countTest += 1

        if ( True == self.flagTest ):
            count = self.countTest
        else:
            count = self.countTrain

        # Draw and save results.
        identifier = "test_%d" % (count - 1)
        loss = self.single_test( identifier, imgL, imgR, disp, self.model, self.criterion )

        # Test the existance of an AccumulatedValue object.
        if ( True == self.frame.have_accumulated_value("lossTest") ):
            self.frame.AV["lossTest"].push_back(loss.item(), self.countTest)
        else:
            self.frame.logger.info("Could not find \"lossTest\"")

        self.frame.plot_accumulated_values()

        return loss.item()

    def single_infer(self, identifier, image0, image1, md):
        """
        identifier: A string identifies this test.
        md:  The pytorch module.
        """
        
        # Forward.
        with torch.no_grad():
            output = md( image0, image1 )
            # output = output[:, :, 4:, :]
        
        # outputTemp = torch.squeeze( output.data.cpu(), 1 )

        # Save the test result.
        batchSize = output.size()[0]
        
        for i in range(batchSize):
            outDisp = output[i, 0, :, :].detach().cpu().numpy()

            outMin = outDisp.min()
            outMax = outDisp.max()

            outDisp = outDisp - outMin

            outDisp = np.clip( outDisp / outMax, 0.0, 1.0 )

            # Create a matplotlib figure.
            fig = plt.figure(figsize=(12.8, 9.6), dpi=300)

            ax = plt.subplot(2, 2, 1)
            plt.tight_layout()
            ax.set_title("Ref")
            ax.axis("off")
            img0 = image0[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img0 = img0 - img0.min()
            img0 = img0 / img0.max()
            plt.imshow( img0 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = image1[i, :, :, :].permute((1,2,0)).cpu().numpy()
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

    def infer(self, imgL, imgR, Q):
        self.check_frame()

        if ( False == self.flagInfer ):
            raise Exception("Not in the infer mode.")

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        self.countTest += 1

        # Draw and save results.
        identifier = "infer_%d" % (self.countTest - 1)
        self.single_infer( identifier, imgL, imgR, self.model )

    # Overload parent's function.
    def finalize(self):
        self.check_frame()
        
        # Save the model.
        if ( False == self.flagTest and False == self.flagInfer ):
            self.frame.save_model( self.model, "CSN" )
