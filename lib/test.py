import pandas as pd
import numpy as np

dataset = np.load('data/npz/hanoi_data.npz')['monitoring_data']
dataset_copy = dataset[:, [-1,-1]].copy()
print(dataset_copy)