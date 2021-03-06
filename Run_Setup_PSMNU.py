
from __future__ import print_function

from workflow import WorkFlow, TorchFlow

import ArgumentParser

from TrainTestPSMNet import TTPSMNU

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range( int(n/2) )]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

# Template for custom WorkFlow object.
class MyWF(TorchFlow.TorchFlow):
    def __init__(self, workingDir, prefix = "", suffix = "", disableStreamLogger=False):
        super(MyWF, self).__init__(workingDir, prefix, suffix, disableStreamLogger)

        # === Custom member variables. ===
        self.tt = None # The TrainTestBase object.

    def set_tt(self, tt):
        self.tt = tt

    def check_tt(self):
        if ( self.tt is None ):
            Exception("self.tt must not be None.")

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        self.check_tt()

        # === Custom code. ===

        self.tt.initialize()

        self.logger.info("Initialized.")

        self.post_initialize()

    # Overload the function train().
    def train(self, imgL, imgR, disp, epochCount):
        super(MyWF, self).train()

        self.check_tt()

        return self.tt.train(imgL, imgR, disp, epochCount)
        
    # Overload the function test().
    def test(self, imgL, imgR, disp, epochCount):
        super(MyWF, self).test()

        self.check_tt()

        return self.tt.test(imgL, imgR, disp, epochCount)

    def infer(self, imgL, imgR, Q):

        self.check_tt()

        self.tt.infer( imgL, imgR, Q )

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        self.check_tt()

        self.tt.finalize()

        self.logger.info("Finalized.")

if __name__ == "__main__":
    print("Hello Setup PSMNU.")

    # Handle the arguments.
    args = ArgumentParser.args

    # Handle the crop settings.
    cropTrain = ArgumentParser.convert_str_2_int_list( args.dl_crop_train )
    cropTest  = ArgumentParser.convert_str_2_int_list( args.dl_crop_test )

    print_delimeter(title = "Before WorkFlow initialization." )

    try:
        # Instantiate an object for MyWF.
        wf = MyWF(args.working_dir, prefix=args.prefix, suffix=args.suffix, disableStreamLogger=False)
        wf.verbose = False

        # Cross reference.
        tt = TTPSMNU(wf.workingDir, wf)
        wf.set_tt(tt)

        if ( True == args.multi_gpus ):
            tt.enable_multi_GPUs()

        if ( True == args.sobel_x ):
            tt.enable_Sobel_x()

        if ( True == args.grayscale ):
            tt.enable_grayscale()

        # tt.flagGrayscale = args.grayscale

        # Set parameters.
        tt.set_learning_rate(args.lr)
        tt.set_max_disparity( args.max_disparity )
        tt.set_data_loader_params( \
            args.dl_batch_size, not args.dl_disable_shuffle, args.dl_num_workers, args.dl_drop_last, \
            cropTrain=cropTrain, cropTest=cropTest )
        tt.set_dataset_root_dir( args.data_root_dir, args.data_entries, args.data_file_list )
        tt.set_read_model( args.read_model )
        tt.enable_auto_save( args.auto_save_model )
        tt.set_training_acc_params( args.train_interval_acc_write, args.train_interval_acc_plot, args.use_intermittent_plotter )

        if ( True == args.test ):
            tt.switch_on_test()
        else:
            tt.switch_off_test()

        if ( True == args.infer ):
            tt.switch_on_infer()
        else:
            tt.switch_off_infer()

        # Initialization.
        print_delimeter(title = "Initialize.")
        wf.initialize()

    except WorkFlow.SigIntException as sie:
        print("SigInt revieved, perform finalize...")
        wf.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")
