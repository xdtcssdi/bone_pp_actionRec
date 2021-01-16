import pandas as pd
import numpy as np
from glob import glob
import argparse
import math
from scipy import signal
parser = argparse.ArgumentParser()
parser.add_argument('--in', type=str, help='no padding csv path', dest='in_')
parser.add_argument('--out', type=str, help='no padding csv path')
args = parser.parse_args()
in_path = args.in_
b, a = signal.butter(8, 0.8, 'lowpass')
df = pd.read_csv(in_path)

for idx in range(36):
    col = df[str(idx)].to_list()
    indice = [i for i, x in enumerate(col) if x==-1]
    indices = []
    start = 0
    end = len(indice)
    if end == 0:
        continue
    for i in range(1, end):
        if indice[i] - indice[i-1] != 1:
            indices.append(indice[start: i])
            start = i
    indices.append(indice[start:])
    for indice in indices:
        next_value = indice[-1] + 1
        if next_value == len(col):
            next_value = col[indice[0]-1]
        else:
            next_value = col[next_value]
        last_value = indice[0] - 1
        if last_value == -1:
            last_value = col[indice[-1]+1]
        else:
            last_value = col[last_value]
        pad_value = np.linspace(last_value,next_value,len(indice))
        for ii, i in enumerate(indice):
            col[i] = pad_value[ii]
        df[str(idx)] = col
    
    col = df[str(idx)].to_list()
    col = signal.filtfilt(b, a, col)
    df[str(idx)] = col
df.to_csv(args.out, index=False)
