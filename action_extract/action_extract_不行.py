import pandas as pd
import csv
import time
import glob
import os
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


## 方差
def judge_with_mv(acc_array, threshold):
    var = np.var(acc_array)
    print(var)
    if var >= threshold:
        return True
    return False


def identify():
    windows_witd = 80
    # 五个动作文件 
    files_path = glob.glob(os.path.join(os.getcwd(), 'bone_padding', '*.csv'))
    files_path.sort()
    print(files_path)
    ths = [0.006, 0.006, 0.006, 0.006, 0.006]  # th的阈值
    cols = None

    for file_, th in zip(reversed(files_path), ths):
        action_file = file_.split(os.sep)[-1]

        cnt = 0  # 提取数量计数器
        dataMat = pd.read_csv(file_, names=['d' +str(i) for i in list(range(36))], low_memory=False)
        d_len = len(dataMat)

        start = 0
        end = windows_witd
        all_action_windows = DataFrame(columns=['start', ].extend(['d' +str(i) for i in list(range(36))]))

        while True:
            if end > d_len:
                break
            windows = dataMat[start:end]
            acc_array = np.array(windows['d8'])
            if judge_with_mv(acc_array, th):
                # 从 acc_array 获取最大值的位置
                max_idx = acc_array.argmax()
                action_windows = windows[max_idx - 20:max_idx + 20]
                
                if len(action_windows) == 40:
                    col = action_windows.values
                    if type(cols) != np.ndarray:
                        cols = col
                    else:
                        cols = np.concatenate((cols, col))
                    #action_windows['start'] = [i for i in range(start + max_idx -20, start + max_idx +20)]
                    all_action_windows = all_action_windows.append(action_windows)
                    cnt += 1
            start += windows_witd //2
            end += windows_witd //2
            
        all_action_windows.to_csv(os.path.join(os.getcwd(),"action_extract", 'proed', action_file))
        print(f"提取到{cnt}个动作")
def draw():
    files_path = glob.glob(os.path.join(dir, '*.csv'))
    files_path.sort()

    in_ = glob.glob(os.path.join(dir, 'proed', '*.csv'))
    in_.sort()

    files_path.extend(in_)

    plt.rcParams['figure.figsize'] = (12.0, 6.0)

    for idx, file_name in enumerate(files_path):
        dataMat = pd.read_csv(file_name,
                              names=['d' + str(i) for i in range(0, 44)],
                              low_memory=False)[1:].drop(['d0'], axis=1).astype("float64")['d43']

        plt.subplot(4, 5, idx + 1)
        plt.plot(list(range(len(dataMat))), dataMat)
        plt.title(file_name.split('\\')[-1])
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    identify()
    # draw()

