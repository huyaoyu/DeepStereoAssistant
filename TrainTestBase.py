from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from workflow import WorkFlow, TorchFlow

from DataLoader.SceneFlow import Loader as DA
from DataLoader import PreProcess
from DataLoader.SceneFlow.utils import list_files_sceneflow_FlyingThings

RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL = 100

class TrainTestBase(object):
    def __init__(self, workingDir, frame=None):
        self.wd = workingDir
        self.frame = frame

        # NN.
        self.countTrain = 0
        self.countTest  = 0

        self.flagGrayscale = False

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

    def initialize(self):
        self.check_frame()

        # Over load these functions if nessesary.
        self.init_workflow()
        self.init_torch()
        self.init_data()
        self.init_model()
        self.post_init_model()
        self.init_optimizer()
    
    def train(self):
        self.check_frame()
    
    def test(self):
        self.check_frame()
    
    def finialize(self):
        self.check_frame()

    def set_frame(self, frame):
        self.frame = frame
    
    def check_frame(self):
        if ( self.frame is None ):
            raise Exception("self.frame must not be None.")
    
    def enable_multi_GPUs(self):
        self.check_frame()

        self.multiGPUs = True

        self.frame.logger.info("Enable multi-GPUs.")

    def set_dataset_root_dir(self, d, nEntries=0):
        self.check_frame()

        if ( False == os.path.isdir(d) ):
            raise Exception("Dataset directory (%s) not exists." % (d))
        
        self.datasetRootDir = d
        self.dataEntries    = nEntries

        self.frame.logger.info("Data root directory is %s." % ( self.datasetRootDir ))
        if ( 0 != nEntries ):
            self.frame.logger.warning("Only %d entries of the training dataset will be used." % ( nEntries ))

    def set_data_loader_params(self, batchSize=2, shuffle=True, numWorkers=2, dropLast=False):
        self.check_frame()

        self.dlBatchSize  = batchSize
        self.dlShuffle    = shuffle
        self.dlNumWorkers = numWorkers
        self.dlDropLast   = dropLast

    def set_read_model(self, readModelString):
        self.check_frame()
        
        self.readModelString = readModelString

        if ( "" != self.readModelString ):
            self.frame.logger.info("Read model from %s." % ( self.readModelString ))

    def enable_auto_save(self, loops):
        self.check_frame()
        
        self.autoSaveModelLoops = loops

        if ( 0 != self.autoSaveModelLoops ):
            self.frame.logger.info("Auto save enabled with loops = %d." % (self.autoSaveModelLoops))

    def set_training_acc_params(self, intervalWrite, intervalPlot, flagInt=False):
        self.check_frame()
        
        self.trainIntervalAccWrite = intervalWrite
        self.trainIntervalAccPlot  = intervalPlot
        self.flagUseIntPlotter     = flagInt

        if ( True == self.flagUseIntPlotter ):
            if ( self.trainIntervalAccPlot <= RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ):
                self.frame.logger.warning("When using the intermittent plotter. It is recommended that the plotting interval (%s) is higher than %d." % \
                    ( self.trainIntervalAccPlot, RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ) )

    def init_workflow(self):
        raise Exception("init_workflow() virtual interface.")

    def init_torch(self):
        self.check_frame()

        self.frame.logger.info("Configure Torch.")

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

    def init_data(self):
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
            batch_size=1, shuffle=False, num_workers=self.dlNumWorkers, drop_last=self.dlDropLast )

    def init_model(self):
        raise Exception("init_model() virtual interface.")

    def post_init_model(self):
        if ( True == self.multiGPUs ):
            self.model = nn.DataParallel(self.model)

        self.model.cuda()
    
    def init_optimizer(self):
        raise Exception("init_optimizer() virtual interface.")
