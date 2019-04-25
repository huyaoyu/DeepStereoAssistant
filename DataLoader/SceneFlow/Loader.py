
from __future__ import print_function

# Most of the content of this file is copied from PSMNet.
# https://github.com/JiaRenChang/PSMNet

import cv2
import numpy as np
from PIL import Image, ImageOps
import random

import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms

from . import IO
from .. import PreProcess 

# import sys

# sys.path.insert(0, "/home/yaoyu/Projects/DeepStereoAssistant/DataLoader")
# import IO
# import PreProcess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def cv2_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def disparity_loader(path):
    return IO.readPFM(path)

class myImageFolder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, \
        loader=cv2_loader, dploader= disparity_loader, preprocessor=None, \
        cropSize=(0,0)):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

        # Modified.
        self.preprocessor  = preprocessor
        self.cropSize = cropSize

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        # import ipdb; ipdb.set_trace()
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:  
            # w, h = left_img.size

            w = left_img.shape[1]
            h = left_img.shape[0]

            if ( self.cropSize[0] <= 0 or self.cropSize[1] <= 0):
                th, tw = h, w
            else:
                # th, tw = 256, 512
                # th, tw = 528, 960
                # th, tw = 256, 960
                th, tw = self.cropSize[0], self.cropSize[1]
 
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            # cv2 compatible crop.
            left_img  = left_img[ y1:y1 + th, x1:x1 + tw ]
            right_img = right_img[ y1:y1 + th, x1:x1 + tw ]

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

        #    processed = PreProcess.get_transform(augment=False)  
        #    left_img   = processed(left_img)
        #    right_img  = processed(right_img)

            if ( self.preprocessor is not None ):
                left_img  = self.preprocessor(left_img)
                right_img = self.preprocessor(right_img)

            return left_img, right_img, dataL
        else:
            # w, h = left_img.size

            w = left_img.shape[1]
            h = left_img.shape[0]

            if ( self.cropSize[0] <= 0 or self.cropSize[1] <= 0 ):
                ch, cw = h, w
            else:
                ch, cw = self.cropSize[0], self.cropSize[1]

            # left_img  = left_img.crop( (w-cw, h-ch, w, h))
            # right_img = right_img.crop((w-cw, h-ch, w, h))

            # cv2 compatible crop.
            left_img  = left_img[ h-ch:h, w-cw:w ]
            right_img = right_img[ h-ch:h, w-cw:w ]

        #    processed = PreProcess.get_transform(augment=False)  
        #    left_img       = processed(left_img)
        #    right_img      = processed(right_img)

            if ( self.preprocessor is not None ):
                left_img  = self.preprocessor(left_img)
                right_img = self.preprocessor(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

class inferImageFolder(data.Dataset):
    def __init__(self, left, right, Q, \
        loader=default_loader, preprocessor=None, \
        cropSize=(0,0)):
 
        self.left   = left
        self.right  = right
        self.Q      = Q
        self.loader = loader

        # Modified.
        self.preprocessor  = preprocessor
        self.cropSize      = cropSize

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        # w, h = left_img.size

        w = left_img.shape[1]
        h = left_img.shape[0]

        if ( self.cropSize[0] <= 0 or self.cropSize[1] <= 0 ):
            ch, cw = h, w
        else:
            ch, cw = self.cropSize[0], self.cropSize[1]

        # left_img  = left_img.crop( (w-cw, h-ch, w, h))
        # right_img = right_img.crop((w-cw, h-ch, w, h))

        # cv2 compatible crop.
        left_img  = left_img[ h-ch:h, w-cw:w ]
        right_img = right_img[ h-ch:h, w-cw:w ]

        if ( self.preprocessor is not None ):
            left_img  = self.preprocessor(left_img)
            right_img = self.preprocessor(right_img)

        # Load the Q matrix.
        Q = self.Q[index]

        return left_img, right_img, torch.from_numpy( np.loadtxt( Q, dtype=np.float32 ) )

    def __len__(self):
        return len(self.left)

if __name__ == "__main__":    
    img = default_loader("/media/yaoyu/DiskE/SceneFlow/Sampler/FlyingThings3D_Manually/frames_cleanpass/TRAIN/A/0000/left/0006.png")
    import ipdb; ipdb.set_trace()

