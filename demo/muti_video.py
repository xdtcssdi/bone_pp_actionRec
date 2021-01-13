import os
import re
import sys
sys.path.append('.')
import cv2, csv
import math
import time
import scipy
import argparse
import matplotlib, os
import numpy as np
import pylab as plt
import torch, numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from glob import glob
from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('--count', type=int,
                    default=10)
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
    filepaths = os.path.join(os.getcwd(), "video_data", "*.mp4");
    for filepath in glob(filepaths):

        # call ffprobe command to get video's total frame number.
        # cal different programe by system platform.
        if sys.platform =='win32':
            program = 'ffprobe.exe'
        else:
            program = 'ffprobe'
        total = os.popen(program + ' -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 ' + filepath)
        total = int(total.read()) // cfg.BATCH_SIZE # calculate total progress
        video_capture = cv2.VideoCapture(filepath) # open video file
        outname = filepath.split(os.sep)[-1].split('.')[0] # extract out filename

        with tqdm(total=total) as pbar:
            while True:
                # extract BATCH_SIZE frames
                batch_imgs = [] # concat array data
                for i in range(cfg.BATCH_SIZE):
                    # Capture frame-by-frame
                    ret, oriImg = video_capture.read()
                    if type(oriImg) != numpy.ndarray:
                        break
                    oriImg = cv2.resize(oriImg, (480, 640))
                    batch_imgs.append(oriImg)
                
                batch_imgs = np.array(batch_imgs)
                # if no frame can extract then break
                if batch_imgs.shape[0] == 0:
                    break

                with torch.no_grad():
                    # deal
                    paf, heatmap, imscale = get_outputs(
                        batch_imgs, model, 'rtpose')

                    # calculate bone points for every heatmap and paf
                    for i in range(batch_imgs.shape[0]):   
                        humans = paf_to_pose_cpp(heatmap[i], paf[i], cfg)

                        # write to csv
                        with open("./action_csv/" + outname + '.csv', 'a') as f:
                            writer = csv.writer(f)
                            
                            data = []
                            for i in range(18):
                                if i in humans[0].body_parts:
                                    item = humans[0].body_parts[i]
                                    data.extend([item.x, item.y])
                                else:
                                    data.extend([-1, -1])

                            writer.writerow(data)

                        # if type(oriImg) != numpy.ndarray:
                        #     break
                        #out = draw_humans(batch_imgs[i], humans)
                # update progress
                pbar.update(1)
        
        # When everything is done, release the capture
        video_capture.release()
