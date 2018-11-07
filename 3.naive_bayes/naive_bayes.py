#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time
import logging
import logging.handlers

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

np.set_printoptions(threshold='nan')

logger = logging.getLogger("naive")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("./log.txt",mode='w')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img) # 二值化处理，高于阈值50，设值为1
    return cv_img

def Train(trainset,train_labels):
    # 需要依据trian数据集，计算先验概率和条件概率
    # 先验概率
    prior_probability = np.zeros(class_num)
    # 条件概率: class_num:Cj(Y的划分空间）；  feature_len：维度p ； 2:p维度的Xi，每个Xi的取值范围（只有0,1）两种取值
    conditional_probability = np.zeros((class_num,feature_len,2))
    print("train_set shape = ", trainset.shape)# train_set shape =  (67, 784)

    # 2.遍历train数据集，开始：计算先验概率及条件概率
    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])  # 图片二值化 , 二值化处理之后，图片变成黑白2色,只有0,1两种值
        label = train_labels[i]
        # 2.1:统计到先验概率（以个数表示，没有换算成概率）
        prior_probability[label] += 1
        # 2.2:label与feature及值的数量统计
        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1 # img[j]为0/1

    # 将概率归到[1.10001]
    for i in range(class_num):
        for j in range(feature_len):

            # 经过二值化后图像只有0，1两种取值
            pix_0 = conditional_probability[i][j][0]
            pix_1 = conditional_probability[i][j][1]

            # 计算0，1像素点对应的条件概率
            probalility_0 = (float(pix_0)/float(pix_0+pix_1))*magnification + 1 # 概率值：全部计算到1.xxxx
            probalility_1 = (float(pix_1)/float(pix_0+pix_1))*magnification + 1 # 方便计算概率时，直接连乘

            conditional_probability[i][j][0] = probalility_0
            conditional_probability[i][j][1] = probalility_1

    return prior_probability,conditional_probability

# 计算概率
def calculate_probability(img,label):
    # 1.1 类别为label（Ck）的数量,认为是（P(Y=Ck))
    probability = int(prior_probability[label])
    for i in range(len(img)):
        # 1.2 在y=label(Ck)条件下，对于每个testimg， X(i)=i的概率,由于概率值已经被归化到1.xxxx，所以可以直接连乘
        probability *= int(conditional_probability[label][i][img[i]]) # trick:img[i]只能为0或1, 最终的值，需要除以maga么？
    return probability

def Predict(testset,prior_probability,conditional_probability):
    predict = []
    for img in testset:
        # 图像二值化
        img = binaryzation(img)

        max_label = 0
        # 1.1 先计算第0个类别C0的概率
        max_probability = calculate_probability(img,0)
        # 1.2 然后计算第1-10个类别Ci的概率，跟P0比较，找到最大的概率
        for j in range(1,10):
            probability = calculate_probability(img,j)
            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)
    return np.array(predict)

class_num = 10
feature_len = 784
magnification = 100 # 1000000
train_num = 3000

if __name__ == '__main__':
    print('Start read data')
    time_1 = time.time()
    # 1.读取数据
    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values
    # 2.切片数据：分img_raw和label
    imgs = data[0:train_num:,1::]
    labels = data[0:train_num:,0]
    # 3.选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    time_2 = time.time()
    print('read data cost ',time_2 - time_1,' second','\n')
    print('Start training')
    # 4.Train: 计算先验概率和条件概率
    prior_probability,conditional_probability = Train(train_features,train_labels)
    time_3 = time.time()
    print('training cost ',time_3 - time_2,' second','\n')

    print("prior = ", prior_probability.tobytes())
    print("condition = ", conditional_probability.tobytes())

    print('Start predicting')
    # 5.预测结果
    test_predict = Predict(test_features,prior_probability,conditional_probability)
    time_4 = time.time()
    print('predicting cost ',time_4 - time_3,' second','\n')
    # 6.查看预测结果
    score = accuracy_score(test_labels,test_predict)
    print("The accruacy socre is ", score)