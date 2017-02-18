import os
from collections import OrderedDict

import numpy as np

import tensorflow as tf

## CONST
## Initialize Constant
flags = tf.app.flags
FLAGS = flags.FLAGS

## Pre-processing Parameters
flags.DEFINE_integer('trans_x_range', 100,
                     'The number of translation pixels up to in the X direction for augmented data (-RANGE/2, RANGE/2).')
flags.DEFINE_integer('trans_y_range', 40,
                     'The number of translation pixels up to in the Y direction for augmented data (-RANGE/2, RANGE/2).')
flags.DEFINE_integer('trans_angle', .3,
                     'The maximum angle change when translating in the X direction')
flags.DEFINE_integer('off_center_img', .25,
                     'The angle change when using off center images')
flags.DEFINE_integer('brightness_range', .25,
                     'The range of brightness changes')
flags.DEFINE_integer('angle_threshold', 1,
                     'The maximum magitude of the angle possible')

## Trainning Parameters
flags.DEFINE_integer('full_epochs', 10,
                     'The number of epochs when end-to-end training.')
flags.DEFINE_integer('features_epochs', 1,
                     'The number of epochs when training features.')
flags.DEFINE_integer('train_batch_per_epoch', 128,
                     'The number of batches per epoch for training')
flags.DEFINE_integer('batch_size', 128, 
                     'The batch size.')

## Nvida's camera format
flags.DEFINE_integer('img_h', 64, 'The image height.')
flags.DEFINE_integer('img_w', 64, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')


## Fix random seed for reproducibility
np.random.seed(42)


## Data path
vid_path = './epochs/epoch01_front.mkv'
img_path = './test_img/'
## 
data_dir = os.path.abspath('./epochs')
out_dir = os.path.abspath('./output')
model_dir = os.path.abspath('./models')
## Split dataset
epochs = OrderedDict()
epochs['train'] = [3, 4, 5, 6, 8]
epochs['val'] = [1, 2, 7, 9]
# epochs['test'] = [10]
