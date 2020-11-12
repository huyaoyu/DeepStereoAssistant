from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description='Train pyramid stereo matching net.')

parser.add_argument("--working-dir", type=str, default="./Debug", \
    help="The working directory.")

parser.add_argument("--input", type=str, default="", \
    help="The filename of the input JSON file.")

parser.add_argument("--read-model", type=str, default="", \
    help="Real model from working directory. Supply empty string for not reading model.")

parser.add_argument("--read-optimizer", type=str, default="", \
    help="Read the optimizer state from the working directory. Leave blank for not reading the optimizer.")

parser.add_argument("--prefix", type=str, default="", \
    help="The prefix of the work flow. The user should supply delimiters such as _ .")

parser.add_argument("--suffix", type=str, default="", \
    help="The suffix o fthe work flow. The user should supply delimiters such as _ .")

parser.add_argument("--multi-gpus", action="store_true", default=False, \
    help="Use multiple GPUs.")

parser.add_argument("--cpu", action="store_true", default=False, \
    help="Set this flag to use cpu only. This will overwrite --multi-gpus flag.")

parser.add_argument("--lr", type=float, default=0.0001, \
    help="Learning rate.")

parser.add_argument("--max-disparity", type=int, default=512, \
    help="The maximum disparity without any scale factor.")

parser.add_argument("--grayscale", action="store_true", default=False, \
    help="Work on grayscale images.")

parser.add_argument("--sobel-x", action="store_true", default=False, \
    help="Work on Sobel filtered images. The filter is applied along x direction. Grayscale image will automatically be used no matter the --grayscale is issued or not.")

parser.add_argument("--dl-batch-size", type=int, default=2, \
    help="The batch size of the dataloader.")

parser.add_argument("--dl-disable-shuffle", action="store_true", default=False, \
    help="The shuffle switch of the dataloader.")

parser.add_argument("--dl-num-workers", type=int, default=2, \
    help="The number of workers of the dataloader.")

parser.add_argument("--dl-resize", type=str, default="0, 0", \
    help="The new h, w. Set \"0, 0\" to disable.")

parser.add_argument("--dl-crop-train", type=str, default="0, 0", \
    help="The the h-crop and w-crop size during training. Set \"0, 0\" to disable.")

parser.add_argument("--dl-crop-test", type=str, default="0, 0", \
    help="The the h-crop and w-crop size during testing. Set \"0, 0\" to disable")

parser.add_argument("--dl-drop-last", action="store_true", default=False, \
    help="The drop-last switch of the dataloader.")

parser.add_argument("--data-root-dir", type=str, default="./Data", \
    help="The root directory of the dataset.")

parser.add_argument("--data-file-list", action="store_true", default=False, \
    help="Use the pre-defined image and disparity list files.")

parser.add_argument("--data-file-list-dir", type=str, default="./Data", \
    help="When the --data-file-list flag is set. Use this argument to specify the directory for the file-lists files.")

parser.add_argument("--data-entries", type=int, default=0, \
    help="Only use the first several entries of the dataset. This is for debug use. Set 0 for using all the data.")

parser.add_argument("--train-epochs", type=int, default=10, \
    help="The number of training epochs.")

parser.add_argument("--test", action="store_true", default=False, \
    help="Only perform test. Make sure to specify --read-model")

parser.add_argument("--test-loops", type=int, default=0, \
    help="The number of training loops between a test. Set 0 for not testing. When perform batch testing, this argument controls the number of test cases.")

parser.add_argument("--test-save-disp", action="store_true", default=False, \
    help="Set this flag to save the disparity when doing a test.")

parser.add_argument("--test-start-index", type=int, default=0, \
    help="The starting index of the test. Only valid under test mode.")

parser.add_argument("--infer", action="store_true", default=False, \
    help="Enable the infer mode.")

parser.add_argument("--infer-save-cloud", action="store_true", default=False, \
    help="Save point cloud when inferring.")

parser.add_argument("--train-interval-acc-write", type=int, default=10, \
    help="Write the accumulated data to filesystem by the number of loops specified.")

parser.add_argument("--train-interval-acc-plot", type=int, default=1, \
    help="Plot the accumulated data to filesystem by the number of loops specified.")

parser.add_argument("--use-intermittent-plotter", action="store_true", default=False, \
    help="Use the intermittent plotter instead of the Visdom plotter. NOTE: Make sure to set --train-interval-acc-plot accordingly.")

parser.add_argument("--auto-save-model", type=int, default=0, \
    help="Plot the number of loops to perform an auto-save of the model. 0 for disable auto-saving.")

parser.add_argument("--disable-stream-logger", action="store_true", default=False, \
    help="Disable the stream logger of WorkFlow.")

parser.add_argument("--inspect", action="store_true", default=False, \
    help="Enable the inspection mode.")

args = parser.parse_args()

def convert_str_2_int_list(s, d=","):
    """
    Convert a string of integers into a list.
    s: The input string.
    d: The delimiter.
    """

    ss = s.split(d)

    temp = []

    for t in ss:
        temp.append( int(t) )

    return temp
