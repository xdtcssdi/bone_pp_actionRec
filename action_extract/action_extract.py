# 导入必要的软件包
import cv2
import os
import csv

# 视频文件输入初始化
camera = cv2.VideoCapture(os.path.join(os.getcwd(), "video_data", "01A.mp4"))

# 初始化当前帧的前两帧
lastFrame1 = None
lastFrame2 = None
start, end = 0, 0
count = 0
is_start = False
larger_100000 = 0
writer = csv.writer(open(os.path.join(os.getcwd(), "action_extract", "data.csv"), 'w', newline=''))
# 遍历视频的每一帧
while camera.isOpened():

    # 读取下一帧
    (ret, frame) = camera.read()
    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not ret:
        break

    # 调整该帧的大小
    frame = cv2.resize(frame, (480, 640), interpolation=cv2.INTER_CUBIC)

    # 如果第一二帧是None，对其进行初始化,计算第一二帧的不同
    if lastFrame2 is None:
        if lastFrame1 is None:
            lastFrame1 = frame
        else:
            lastFrame2 = frame
            global frameDelta1  # 全局变量
            frameDelta1 = cv2.absdiff(lastFrame1, lastFrame2)  # 帧差一
        continue

    # 计算当前帧和前帧的不同,计算三帧差分
    frameDelta2 = cv2.absdiff(lastFrame2, frame)  # 帧差二
    thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算
    
    # 如果帧差小于100000
    # 那么动作静止, end = 当前帧
    # 否则在运动中，第一次检测在运动中时设置start = 当前帧
    # 
    if thresh.sum() > 100000:
        larger_100000 += 1
        if larger_100000 == 20:
            larger_100000 = 0
            # 动作开始
            end = count
            is_start = True
    else:
        if is_start:
            # 检测到动作
            # 进行处理
            writer.writerow([start, end])
        is_start = False
        start = count
        larger_100000 = 0

    
    # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
    lastFrame1 = lastFrame2
    lastFrame2 = frame.copy()
    frameDelta1 = frameDelta2

    # 显示当前帧
    cv2.imshow("frame", frame)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1

# 清理资源并关闭打开的窗口
camera.release()
cv2.destroyAllWindows()