import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import  accuracy_score

k = 10
if __name__ == '__main__':
    print("start.....")
    raw_data = pd.read_csv('../data/train.csv',header=0)
    print("raw_data.shape = ",raw_data.shape)
    data = raw_data.values
    print("data shape = ", data.shape)
    imgs = data[0::,1::]
    labels = data[::,0]
    print("imgs.shape = ", imgs.shape)
    print("labels.shape = ", labels.shape)

