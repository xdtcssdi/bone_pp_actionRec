import os
import re
import sys
import csv
sys.path.append('.')
import cv2, csv
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp


parser = argparse.ArgumentParser()

parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)

parser.add_argument('--weight', type=str,
                    default='pose_model.pth')

parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)   

model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model.cuda()
model.float()
model.eval()

if __name__ == "__main__":
    
    cap = cv2.VideoCapture(os.path.join(os.getcwd(), "video_data", "01A.mp4"))
    
    count = 0
    frame_idx = 0
    #pose = list(csv.reader(open(os.path.join(os.getcwd(), "action_csv", "01A.csv"))))
    #pose = list(csv.reader(open(os.path.join(os.getcwd(), "bone_padding", "01A_pad.csv"))))
    # 输入的是带表头的数据
    pose = list(csv.reader(open(os.path.join(os.getcwd(), "action_extract", "proed", "01A_pad.csv"))))[1:]
    while cap.isOpened():
        
        ret, oriImg = cap.read()
        oriImg = cv2.resize(oriImg, (480, 640))
        oriImg = oriImg[np.newaxis, :]
        
        humans = [Human([])]
        row = pose[frame_idx]
        if int(row[0]) != count + 1:
            #print(int(row[-1]), count)
            count+=1
            continue
        frame_idx += 1
        count+=1
        
        for i in range(18):
            x, y = float(row[i*2+1]), float(row[i*2+2])
            if x!=-1 and y!=-1:
                bodyPart = BodyPart(0, i, x, y, 1)
                humans[0].body_parts[i]= bodyPart
                
        out = draw_humans(oriImg[0], humans)
        time.sleep(0.1)
        # Display the resulting frame
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    #video_capture.release()
    cv2.destroyAllWindows()
