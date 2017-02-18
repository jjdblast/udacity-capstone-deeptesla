#!/usr/bin/env python 
import sys
import os
import time
import subprocess as sp
import itertools

import cv2
import tensorflow as tf

import params
import preprocess
import visualize
import video_control
import model

## Test epoch
epoch_ids = [10]
## Load model
model = model.get_model('best_vgg')

## Process video
for epoch_id in epoch_ids:
    print('---------- processing video for epoch {} ----------'.format(epoch_id))
    vid_path = video_control.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(vid_path)
    frame_count = video_control.frame_count(vid_path)
    cap = cv2.VideoCapture(vid_path)

    machine_steering = []

    print('performing inference...')
    time_start = time.time()
    for frame_id in range(frame_count):
        ret, img = cap.read()
        assert ret
        ## you can modify here based on your model
        img = preprocess.img_pre_process(img)
        img = img[None,:,:,:]
        deg = float(model.predict(img, batch_size=1))
        machine_steering.append(deg)

    cap.release()

    fps = frame_count / (time.time() - time_start)
    
    print('completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1)))
    
    print('performing visualization...')
    visualize.visualize(epoch_id, machine_steering, params.out_dir,
                        verbose=True, frame_count_limit=None)
    
    
