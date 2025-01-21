import os
import cv2 as cv
import importlib
import os
import time
from collections import OrderedDict
from pathlib import Path
import sys
import numpy as np

from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.utils.lmdb_utils import decode_img
from lib.test.evaluation.tracker import Tracker
from lib.utils.load import load_yaml

import argparse
parser =argparse.ArgumentParser()
parser.add_argument("video")
args = parser.parse_args()
tracker = Tracker(
    name='lightfc', parameter_name="lightfc_datasets", dataset_name='OTB2015')
# video_path = '/home/wen/Downloads/1.2.mp4'
# video_path = '/home/wen/Documents/1_1_20250114_大疆neo无人机-环绕画面_20250114_大疆neo无人机-环绕画面.mp4'
video_path = args.video

params = TrackerParams()
# yaml_file = '/home/devon/Project/TRACKER/SOT/LightFC/experiments/lightfc/mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou.yaml'
# params.cfg = load_yaml(yaml_file)
# params.template_factor = 2.0
# params.template_size = 128
# params.search_factor =  4.0
# params.search_size = 256
# params.checkpoint = '/home/devon/Project/TRACKER/SOT/LightFC/output/checkpoints/train/lightfc/mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou/lightfc_ep0400.pth.tar'

yaml_file = '/home/wen/LightFC/experiments/lightfc/mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou.yaml'
params.cfg = load_yaml(yaml_file)
params.template_factor = 2.0
params.template_size = 128
params.search_factor =  4.0
params.search_size = 256
params.checkpoint = '/home/wen/LightFC/output/checkpoints/train/lightfc/mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou/lightfc_ep0400.pth.tar'


light_tracker =tracker.create_tracker(params)

def _build_init_info(box):
    return {'init_bbox': box}

cap = cv.VideoCapture(video_path)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv.CAP_PROP_FPS))
print('frame_width:', frame_width)
print('frame_height:', frame_height)
print('frame_rate:', frame_rate)
cap.set(cv.CAP_PROP_POS_FRAMES, 80 * 30)

writer = cv.VideoWriter('output_lightfc-vit.avi', cv.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))

count = 20
while count:
    ret, frame = cap.read()
    count = count-1
x, y, w, h  = cv.selectROI('Video', frame)
# x, y, w, h = 1619, 181, 52, 51

init_state = [x, y, w, h]
light_tracker.initialize(frame, _build_init_info(init_state))

print('Start tracking...')

while True:
    ret, frame = cap.read()

    if frame is None:
        break

    frame_disp = frame.copy()

    # Draw box
    out = light_tracker.track(frame)
    state = [int(s) for s in out['target_bbox']]

    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                     (0, 255, 0), 2)

    font_color = (0, 0, 0)
    cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                font_color, 1)
    cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                font_color, 1)
    cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                font_color, 1)
    
    writer.write(frame_disp)

    # Display the resulting frame
    cv.imshow('Video', frame_disp)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

