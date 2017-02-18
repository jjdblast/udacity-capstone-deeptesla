## Sys
import os
import sys
import re
import datetime
import time
import copy
import subprocess as sp
## Data structure
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import six
## CV
import cv2
## Model
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
## Parameters
import params ## you can modify the content of params



#########################################
#### Preprocess
#########################################
def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
    ## Resize the image
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    ## Return the image sized as a 4D array
    return np.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h, params.FLAGS.img_c))


#########################################
#### Video control
#########################################
## Basic function
def is_sequence(arg):
    return (not hasattr(arg, 'strip') and 
            hasattr(arg, '__getitem__') and
            hasattr(arg, '__iter__'))

def is_int(s):
    assert not is_sequence(s)
    try: 
        int(s)
        return True
    except ValueError:
        return False

## Return data path for loader
def join_dir(dirpath, filename):
    return os.path.join(dirpath, filename)    

## Load csv file
def fetch_csv_data(path):
    return pd.read_csv(path)

## Load video file
def imread(img_path, mode=cv2.IMREAD_COLOR):
    assert os.path.isfile(img_path), 'Bad image path: {}'.format(img_path)
    return cv2.imread(img_path, mode)

def cv2_current_frame(cap):
    x = cap.get(cv2.CAP_PROP_POS_FRAMES)
    assert x.is_integer()
    return int(x)

def cv2_goto_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    assert cv2_current_frame(cap) == frame_id

##
def frame_count(path, method='ffmpeg'):
    if method == 'ffmpeg':
        return ffmpeg_frame_count(path)
    else:
        assert False 

##     
def ffmpeg_frame_count(path):
    cmd = 'ffmpeg -i {} -vcodec copy -acodec copy -f null /dev/null 2>&1'.format(path)
    cmd_res = sp.check_output(cmd, shell=True)
    cmd_res = copy.deepcopy(cmd_res)

    fc = None

    lines = cmd_res.splitlines()
    lines = lines[::-1]

    for line in lines:
        line = line.strip()
        # res = re.match(r'frame=\s*(\d+)\s*fps=', line)
        res = re.match(b'frame=\s*(\d+)\s*fps=', line)
        if res:
            fc = res.group(1)
            
            assert is_int(fc)
            fc = int(fc)
            break

    assert fc is not None

    return fc
##
def without_ext(path): 
    return os.path.splitext(path)[0]

def ext(path, period=False):
    x = os.path.splitext(path)[1]
    x = x.replace('.', '')
    return x

def mkv_to_mp4(mkv_path, remove_mkv=False):
    assert os.path.isfile(mkv_path)
    assert ext(mkv_path) == 'mkv'
    mp4_path = without_ext(mkv_path) + '.mp4'
    
    if os.path.isfile(mp4_path):
        os.remove(mp4_path)
    
    cmd = 'ffmpeg -i {} -c:v copy -c:a libfdk_aac -b:a 128k {} >/dev/null 2>&1'.format(mkv_path, mp4_path)
    sp.call(cmd, shell=True)

    assert os.path.isfile(mp4_path) # make sure that the file got generated successfully

    if remove_mkv:
        assert os.path.isfile(mkv_path)
        os.remove(mkv_path)
##
def video_resolution_to_size(resolution, width_first=True):
    if resolution == '720p':
        video_size = (1280, 720)
    elif resolution == '1080p':
        video_size = (1920, 1080)
    elif resolution == '1440p':
        video_size = (2560, 1440)
    elif resolution == '4k':
        video_size = (3840, 2160)
    else: assert False

    if not width_first:
        video_size = (video_size[1], video_size[0])
    return video_size
    
def cv2_resize_by_height(img, height):
    ratio = height / img.shape[0]
    width = ratio * img.shape[1]
    height, width = int(round(height)), int(round(width))
    return cv2.resize(img, (width, height))
              
## Video output
def overlay_image(l_img, s_img, x_offset, y_offset):
    assert y_offset + s_img.shape[0] <= l_img.shape[0]
    assert x_offset + s_img.shape[1] <= l_img.shape[1]

    l_img = l_img.copy()
    for c in range(0, 3):
        l_img[y_offset:y_offset+s_img.shape[0],
              x_offset:x_offset+s_img.shape[1], c] = (
                  s_img[:,:,c] * (s_img[:,:,3]/255.0) +
                  l_img[y_offset:y_offset+s_img.shape[0],
                        x_offset:x_offset+s_img.shape[1], c] *
                  (1.0 - s_img[:,:,3]/255.0))
    return l_img


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape)/2)[:2]
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
    return result

#########################################
#### Model load
#########################################
def save_model(model, epoch=''):
    """
    Saves the model and the weights to a json file
    :param model: The mode to be saved
    :param epoch: The epoch number, so as to save the model to a different file name after each epoch
    :return: None
    """
    model_path = video_control.join_dir(params.model_dir, 'model_{}.json'.format(epoch))
    param_path = video_control.join_dir(params.model_dir, 'model_{}.h5'.format(epoch))
    #
    json_string = model.to_json()

    with open(model_path, 'w') as outfile:
        outfile.write(json_string)
    model.save_weights(param_path)
    print('Model saved')
    
def get_model(epoch):
    """
    Defines the model
    :return: Returns the model
    """
    """
    Check if a model already exists
    """
    model_path = video_control.join_dir(params.model_dir, 'model_{}.json'.format(epoch))
    param_path = video_control.join_dir(params.model_dir, 'model_{}.h5'.format(epoch))
    
    if os.path.exists(model_path):
        ch = input('Model'+str(epoch)+' already exists, do you want to reuse? (y/n): ')
        if ch == 'y' or ch == 'Y':
            with open(model_path, 'r') as in_file:
                json_model = in_file.read()
                model = model_from_json(json_model)

            weights_file = os.path.join(param_path)
            model.load_weights(weights_file)
            print('Model' +str(epoch)+ ' fetched from the disk')
            model.summary()
    return model



