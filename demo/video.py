import os
import re
import sys
import csv
sys.path.append('.')
import cv2
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
    
    cap = cv2.VideoCapture(os.path.join(os.getcwd(), "video_data/01A.mp4"))
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, oriImg = cap.read()
        
        shape_dst = np.min(oriImg.shape[0:2])

        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')
         
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        if len(humans) !=0:
            
            with open('01A001' + '.csv', 'a') as f:
                writer = csv.writer(f)
                
                data = []
                for i in range(19):
                    if i in humans[0].body_parts:
                        item = humans[0].body_parts[i]
                        data.extend([item.x, item.y, item.score])
                    else:
                        data.extend([-1, -1, -1])

                writer.writerow(data)
        out = draw_humans(oriImg, humans)

        # Display the resulting frame
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    #video_capture.release()
    cv2.destroyAllWindows()
