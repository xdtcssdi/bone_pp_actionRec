# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from glob import glob


# %%
for file_path in glob('./action_train/*.csv'):
    df = pd.read_csv(file_path, names=list(range(57))).drop([54, 55, 56], 1)
    for idx in range(54):
        col = df[idx].to_list()
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
            df[idx] = col
    df.to_csv(file_path)


