# 只检测动作开始
import cv2
import os
import csv
import torch
import numpy as np
import sys
import argparse
sys.path.append('.')
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.network.rtpose_vgg import get_model
from lib.config import update_config, cfg

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

def img2bone(oriImg):
    oriImg = oriImg[np.newaxis, :]
    with torch.no_grad():
        paf, heatmap, imscale = get_outputs(
            oriImg, model, 'rtpose')
    human = paf_to_pose_cpp(heatmap[0], paf[0], cfg)
    return human


def img2bone2(oriImg):
    with torch.no_grad():
        paf, heatmap, imscale = get_outputs(
            oriImg, model, 'rtpose')
        humans = []
        for i in range(oriImg.shape[0]):
            human = paf_to_pose_cpp(heatmap[i], paf[i], cfg)
            humans.append(human)
    return humans


# 视频文件输入初始化
camera = cv2.VideoCapture(cfg.FILE)
out_fps = 10.0  # 输出文件的帧率
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
# 初始化当前帧的前两帧
lastFrame1 = None
lastFrame2 = None
start, end = 0, 0
count = 0
is_start = False
larger_100000 = 0
seq_count = 0
action_count = 0
# 遍历视频的每一帧
def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    return binary

# 创建存储目录文件夹
if cfg.FILE != 0:
    filename = cfg.FILE.split(os.sep)[-1].split('.')[0]
else:
    filename = "capture"
if not os.path.exists('action_video'):
    os.mkdir('action_video')
if not os.path.exists(os.path.join('action_video', 'pose')):
    os.mkdir(os.path.join('action_video', 'pose'))
if not os.path.exists(os.path.join('action_video', 'raw')):
    os.mkdir(os.path.join('action_video', 'raw'))
if not os.path.exists(os.path.join('action_video', 'csv')):
    os.mkdir(os.path.join('action_video', 'csv'))
if not os.path.exists(os.path.join('action_video', 'pose', filename)):
    os.mkdir(os.path.join('action_video', 'pose', filename))
if not os.path.exists(os.path.join('action_video', 'raw', filename)):
    os.mkdir(os.path.join('action_video', 'raw', filename))
if not os.path.exists(os.path.join('action_video', 'csv', filename)):
    os.mkdir(os.path.join('action_video', 'csv', filename))
raw_video = []
pose_video = []
csv_data = []
right_hand_track = []
stop_count = 0
start_frame, end_frame = None, None
start_count = 0
while camera.isOpened():

    # 读取下一帧
    (ret, frame) = camera.read()
   
    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not ret:
        break
    frame = np.rot90(np.rot90(np.rot90(frame)))
    # 调整该帧的大小
    frame = cv2.resize(frame, (480, 640), interpolation=cv2.INTER_CUBIC)
    humans = img2bone(frame)
    if len(humans) == 0:
        continue
    
    raw_video.append(frame)
    out = draw_humans(frame.copy(), humans)
    pose_video.append(out)
    data = []
    for i in range(18):
        if i in humans[0].body_parts:
            item = humans[0].body_parts[i]
            data.extend([item.x, item.y])
        else:
            data.extend([-1, -1])
    csv_data.append(data)


    right_hand_track.append(humans[0].body_parts) 
    if len(right_hand_track) < 3:       
        continue
    last1, last2, this_frame = right_hand_track[-3:]

    c = 0 # 3帧位置差
    for i in range(18):
        if i in [14, 15, 16, 17, 0, 1]:
            continue
        if i in last1 and i in last2:
            c_x = abs(last1[i].x-last2[i].x)+abs(last1[i].x-last2[i].x)
            c_y = abs(last1[i].y-last2[i].y)+abs(last1[i].y-last2[i].y)
            c += c_x + c_y

    for i in range(18):
        if i in [14, 15, 16, 17, 0, 1]:
            continue
        if i in last2 and i in this_frame:
            c_x = abs(last2[i].x-this_frame[i].x)+abs(last2[i].x-this_frame[i].x)
            c_y = abs(last2[i].y-this_frame[i].y)+abs(last2[i].y-this_frame[i].y)
            c += c_x + c_y
    print(c)
    if c > 0.2:
        if not is_start:
            start_count += 1
            if start_count == 8:
                start_count = 0
                is_start = True
                start_frame = count - 8
                print(f"开始{count}")
    elif c < 0.1:
        if is_start:
            stop_count += 1
            if stop_count == 5:
                stop_count = 0
                is_start = False
                end_frame = count - 5
                print(start_frame, end_frame)
                save_action_pose = cv2.VideoWriter(os.path.join('action_video', 'pose', filename, str(action_count)+".avi"), fourcc, out_fps, (480, 640))
                save_action_raw = cv2.VideoWriter(os.path.join('action_video', 'raw', filename, str(action_count)+".avi"), fourcc, out_fps, (480, 640))
                writer = csv.writer(open(os.path.join('action_video', 'csv', filename, str(action_count)+".csv"), 'w', newline=''))
                writer.writerow(['frame'] + list(range(36)))
                # 一次处理一个帧
                for idx, (raw, pose, csv_) in enumerate(zip(raw_video[start_frame:end_frame], pose_video[start_frame:end_frame], csv_data[start_frame:end_frame])):
                    writer.writerow([start_frame + idx, ] + csv_)
                    save_action_raw.write(raw)
                    save_action_pose.write(pose)
                
                action_count += 1
    # if c > 0.3 or is_start:
    #     # 检测到动作
    #     if not is_start:
    #         start = count
    #         is_start = True
    #         # 创建文件写入
    #         save_action_pose = cv2.VideoWriter(os.path.join('action_video', 'pose', filename, str(
    #             action_count)+".avi"), fourcc, out_fps, (480, 640))
    #         save_action_raw = cv2.VideoWriter(os.path.join('action_video', 'raw', filename, str(
    #             action_count)+".avi"), fourcc, out_fps, (480, 640))
    #         writer = csv.writer(open(os.path.join(
    #             'action_video', 'csv', filename, str(action_count)+".csv"), 'w', newline=''))
    #         writer.writerow(['frame'] + list(range(36)))
            
    #         action_count += 1
    #         print(f"检测到第{action_count}个动作")

    #         writer.writerow([start + idx, ] + data)
    #         save_action_raw.write(frame)
    #         save_action_pose.write(out)

        # 一次处理一个帧
        # for idx, frame in enumerate(frames[start: end+1]):
        #     humans = img2bone(frame)
        #     if len(humans) == 0:
        #         continue
        #     data = []
        #     for i in range(18):
        #         if i in humans[0].body_parts:
        #             item = humans[0].body_parts[i]
        #             data.extend([item.x, item.y])
        #         else:
        #             data.extend([-1, -1])
        #     writer.writerow([start + idx, ] + data)
        #     save_action_raw.write(frame)
        #     out = draw_humans(frame, humans)
        #     save_action_pose.write(out)

        # 一次处理所有帧
        # humans = img2bone2(np.array(frames[start: end+1]))

        # for idx, (human, frame) in enumerate(zip(humans, frames[start: end+1])):
        #     data = []
        #     for i in range(18):
        #         if i in human[0].body_parts:
        #             item = human[0].body_parts[i]
        #             data.extend([item.x, item.y])
        #         else:
        #             data.extend([-1, -1])
        #     writer.writerow([start + idx, ] + data)
        #     save_action_raw.write(frame)
        #     out = draw_humans(frame, human)
        #     save_action_pose.write(out)


    # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一

    # 显示当前帧
    
    cv2.imshow("frame", out)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1

# 清理资源并关闭打开的窗口
camera.release()
cv2.destroyAllWindows()
