import csv
import pandas as pd
import numpy as np
frame_data = list(csv.reader(open('data.csv', 'r')))
pose_data = pd.read_csv('bone_padding//01A_pad.csv')
print(pose_data.head())
data = []
for item in frame_data:
    start, end = [int(i) for i in item]
    df = pose_data[start:end+1]
    if type(data) == list:
        data = df.values
    else:
        data = np.concatenate([data, df.values])
pd.DataFrame(data, columns= ['frame', ] + [i for i in range(36)]).to_csv("action_extract/01A_pad2.csv",index=False)