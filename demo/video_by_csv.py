import os
import re
import sys
import csv
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pandas as pd
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter
sys.path.append('.')
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

if __name__ == "__main__":
    
    cap = cv2.VideoCapture(os.path.join("D:\\code\\video_data\\2021-1-19lhd（03）", "03D.mp4"))
    
    count = 0
    frame_idx = 0
    #pose = list(csv.reader(open(os.path.join(os.getcwd(), "action_csv", "01A.csv"))))
    pose = pd.read_csv(os.path.join(os.getcwd(), "padding_csv", "03D.csv")).values

    print(pose.shape)
    # 输入的是带表头的数据
    #pose = list(csv.reader(open(os.path.join(os.getcwd(), "action_extract", "proed", "01A_pad2.csv"))))[1:]
    while cap.isOpened():
        
        ret, oriImg = cap.read()
        if not ret:
            break
        oriImg = np.rot90(np.rot90(np.rot90(oriImg)))
        oriImg = cv2.resize(oriImg, (480, 640))
        
        oriImg = oriImg[np.newaxis, :]
        
        humans = [Human([])]
        row = pose[frame_idx]
        print(row)
        if int(float(row[0])) != count:
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

        cv2.imshow('Video', out)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
