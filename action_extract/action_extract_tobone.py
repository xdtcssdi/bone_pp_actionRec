# 导入必要的软件包
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
frames = []
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
while camera.isOpened():

    # 读取下一帧
    (ret, frame) = camera.read()
   
    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not ret:
        break
    
    # 调整该帧的大小
    frame = cv2.resize(frame, (480, 640), interpolation=cv2.INTER_CUBIC)
    raw_video.append(frame)
    #frames.append(frame)

    humans = img2bone(frame)
    out = draw_humans(frame.copy(), humans)
    pose_video.append(out)
    if len(humans) == 0:
        continue
    data = []
    for i in range(18):
        if i in humans[0].body_parts:
            item = humans[0].body_parts[i]
            data.extend([item.x, item.y])
        else:
            data.extend([-1, -1])
    csv_data.append(data)
    

    # 如果第一二帧是None，对其进行初始化,计算第一二帧的不同
    if lastFrame2 is None:
        if lastFrame1 is None:
            lastFrame1 = local_threshold(frame)
        else:
            lastFrame2 = local_threshold(frame)
            global frameDelta1  # 全局变量
            frameDelta1 = cv2.absdiff(lastFrame1, lastFrame2)  # 帧差一
        continue

    # 计算当前帧和前帧的不同,计算三帧差分
    frameDelta2 = cv2.absdiff(lastFrame2, local_threshold(frame))  # 帧差二
    thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算

    # 如果帧差小于100000
    # 那么动作静止, end = 当前帧
    # 否则在运动中，第一次检测在运动中时设置start = 当前帧
    #
    c = thresh.sum()
    #print(c)
    # 150000 apple
    # 300000 xiaomi
    # 1300000 logi
    if c > 1300000:
        larger_100000 += 1
        if larger_100000 == 20:
            larger_100000 = 0
            # 动作开始
            end = count
            is_start = True
    else:
        if is_start:
            # 检测到动作

            # 创建文件写入
            save_action_pose = cv2.VideoWriter(os.path.join('action_video', 'pose', filename, str(
                action_count)+".avi"), fourcc, out_fps, (480, 640))
            save_action_raw = cv2.VideoWriter(os.path.join('action_video', 'raw', filename, str(
                action_count)+".avi"), fourcc, out_fps, (480, 640))
            writer = csv.writer(open(os.path.join(
                'action_video', 'csv', filename, str(action_count)+".csv"), 'w', newline=''))
            writer.writerow(['frame'] + list(range(36)))
            
            action_count += 1
            print(f"检测到第{action_count}个动作")

            for idx, (raw, pose, csv_) in enumerate(zip(raw_video[-20:], pose_video[-20:], csv_data[-20:])):
                writer.writerow([start + idx, ] + csv_)
                save_action_raw.write(raw)
                save_action_pose.write(pose)

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

        is_start = False
        start = count
        larger_100000 = 0

    # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
    lastFrame1 = lastFrame2
    lastFrame2 = local_threshold(frame.copy())
    frameDelta1 = frameDelta2

    # 显示当前帧
    
    cv2.imshow("frame", out)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1

# 清理资源并关闭打开的窗口
camera.release()
cv2.destroyAllWindows()
