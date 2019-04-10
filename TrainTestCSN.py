from __future__ import print_function

from workflow import WorkFlow, TorchFlow

from TrainTestBase import TrainTestBase

from DataLoader.SceneFlow import Loader as DA
from DataLoader import PreProcess
from DataLoader.SceneFlow.utils import list_files_sceneflow_FlyingThings
from model.CSN.ConvolutionalStereoNet import ConvolutionalStereoNet

import os
import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("TrainTestCSN: Environment variable DISPLAY is not present in the system.")
    print("TrainTestCSN: Switch the backend of matplotlib to agg.")

class TTCSN(TrainTestBase):
    def __init__(self, workingDir, frame=None):
        super(TTCSN, self).__init__( workingDir, frame )

        self.params = params
        self.criterion = torch.nn.SmoothL1Loss()

    # def initialize(self):
    #     self.check_frame()
    #     Exception("Not implemented.")

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
    #     Exception("Not implemented.")

    # def init_torch(self):
    #     Exception("Not implemented.")

    # def init_data(self):
    #     Exception("Not implemented.")

    # Overload parent's function.
    def init_model(self):
        # Neural net.
        if ( True == self.flagGrayscale ):
            Exception("Grayscale is not implemented for CSN yet.")
        else:
            self.model = ConvolutionalStereoNet()

        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                Exception("Model file (%s) does not exist." % ( modelFn ))

            self.model = self.frame.load_model( self.model, modelFn )

        self.frame.logger.info("CSN has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     Exception("Not implemented.")

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        self.optimizer = optim.Adam( self.model.parameters(), lr=self.params["torchOptimLearningRate"] )

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
        loss   = cri( output, disparity0 )

        # Handle the loss value.
        self.frame.AV["loss"].push_back( loss.item() )

        # Backward.
        loss.backward()
        opt.step()

    # Overload parent's function.
    def train(self, imgL, imgR, disp, epochCount):
        self.check_frame()

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
        
        loss   = cri( output, disparity0 )

        # Handle the loss value.
        plotX = self.countTrain - 1
        if ( plotX < 0 ):
            plotX = 0
        self.frame.AV["lossTest"].push_back( loss.item(), plotX )

        # Save the test result.
        batchSize = output.size()[0]
        
        for i in range(batchSize):
            outDisp = output[i, 0, :, :].detach().cpu().numpy()
            gdtDisp = disparity0[i, 0, :, :].detach().cpu().numpy()

            outDisp = outDisp - outDisp.min()
            gdtDisp = gdtDisp - outDisp.min()

            outDisp = outDisp / outDisp.max()
            gdtDisp = gdtDisp / gdtDisp.max()

            # Create a matplotlib figure.
            fig = plt.figure()

            ax = plt.subplot(2, 1, 1)
            plt.tight_layout()
            ax.set_title("Ground truth")
            ax.axis("off")
            plt.imshow( gdtDisp )

            ax = plt.subplot(2, 1, 2)
            plt.tight_layout()
            ax.set_title("Prediction")
            ax.axis("off")
            plt.imshow( outDisp )

            figName = "%s_%02d" % (identifier, i)
            figName = self.frame.compose_file_name(figName, "png")
            plt.savefig(figName)

            plt.close(fig)

        return loss

    # Overload parent's function.
    def test(self, imgL, imgR, disp, epochCount):
        self.check_frame()

        self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        # mask = disp < 192

        identifier = "test_%d" % (self.countTrain - 1)
        loss = self.single_test( identifier, imgL, imgR, disp, self.model, self.criterion )

        self.countTest += 1

        # Test the existance of an AccumulatedValue object.
        if ( True == self.frame.have_accumulated_value("lossTest") ):
            self.frame.AV["lossTest"].push_back(loss.item(), self.countTest)
        else:
            self.frame.logger.info("Could not find \"lossTest\"")

        self.frame.plot_accumulated_values()

        return loss.item()

    # Overload parent's function.
    def finalize(self):
        self.check_frame()
        
        # Save the model.
        self.frame.save_model( self.model, "CSN" )
