from __future__ import print_function

from workflow import WorkFlow, TorchFlow

from TrainTestBase import TrainTestBase

from DataLoader.SceneFlow import Loader as DA
from DataLoader import PreProcess
from DataLoader.SceneFlow.utils import list_files_sceneflow_FlyingThings
from model import PyramidNet

class TTPSMNet(TrainTestBase):
    def __init__(self, workingDir, frame=None):
        super(TTPSMNet, self).__init__( workingDir, frame )

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
            self.model = PyramidNet.PSMNet(1, 32, 64)
        else:
            self.model = PyramidNet.PSMNet(3, 32, 64)

        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                Exception("Model file (%s) does not exist." % ( modelFn ))

            self.model = self.frame.load_model( self.model, modelFn )

        self.frame.logger.info("PSMNet has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     Exception("Not implemented.")

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        self.optimizer = optim.Adam( self.model.parameters(), lr=0.001 )

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
                modelName = "AutoSave_%06d" % ( self.countTrain )
                self.frame.logger.info("Auto-save the model.")
                self.frame.save_model( self.model, modelName )

        self.frame.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.frame.get_log_str()))

    # Overload parent's function.
    def test(self, imgL, imgR, disp, epochCount):
        self.check_frame()

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
        self.frame.save_model( self.model, "PSMNet" )
