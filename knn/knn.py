#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = [] # feature为py中的list
    # 通过参数，构造HOGDescriptor对象，指定winSize,blockSize,cellSize, nbins
    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28)) # 28*28 = 784
        cv_img = img.astype(np.uint8) # 类型转换：

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features) # 将list转换在矩阵
    features = np.reshape(features,(-1,324)) # 324 的feature如何提取？

    return features

def Predict(testset,trainset,train_labels):
    predict = [] # 预测结果
    count = 0    # test集中的预测计数,方便查看进度

    for test_vec in testset:
        # 输出当前运行的测试用例坐标，用于测试
        print("count = ", count)
        count += 1 # 额，进度好慢。。。

        knn_list = []       # 当前k个最近邻居
        max_index = -1      # 当前k个最近邻居中距离最远点的坐标
        max_dist = 0        # 当前k个最近邻居中距离最远点的距离

        # 先将前k个点放入k个最近邻居中，填充满knn_list
        for i in range(k):
            train_vec = trainset[i] # 前k个 图片
            label = train_labels[i] # 前k个 label
            # 计算当前的test_vec，跟前k个点（随机选中的k个中心），每个点的距离
            dist = np.linalg.norm(train_vec - test_vec)  # 计算两个点的欧氏距离

            knn_list.append((dist,label)) # 并保留K个邻居的  dist和label

        # trainset中，除去前K个点后，剩下的点
        for i in range(k,len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]
            # 剩下的点中，计算test_vec跟每个点的距离
            # test_vec 与当前点的距离
            dist = np.linalg.norm(train_vec - test_vec)  # 计算两个点的欧氏距离

            # 寻找10个邻近点中距离最远的点 ： 距离当前的test_vec最远的点
            # 后续点的计算中，计算完当前点跟test_vec之间的距离，遍历查询 knn_list
            # 查看test_vec 与当前点的距离， 及test_vec与前K个点的距离，找到最大的那个点的下标（max_index）和距离（max_dist）
            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]

            # 如果当前k个最近邻居中存在点距离比当前点距离远，则替换
            # test_vec 跟当前点的距离，比之前我们随机给他指定的K个邻居中的某个点的距离小，那么先将 距离最大的那个邻居 “置换”出来，
            # 相当于：前K个随机指定的邻居中，有邻居不合格，当前点更适合做test_vec的邻居
            if dist < max_dist:
                # 将不合格的邻居置换出来
                knn_list[max_index] = (dist,label)
                # 然后重置标记位
                max_index = -1
                max_dist = 0

        # 对当前test_vec的计算结束，统计选票
        # 此时，完成test_set中的这1个test_vec向量，与train_set（带标记的向量）中每个点的距离计算
        # 并且，已经找到了test_vec 距离最近的K个邻居
        class_total = 10  # 还是使用常量K吧
        class_count = [0 for i in range(class_total)]
        for dist,label in knn_list:
            class_count[label] += 1 #统计K个邻居中，属于哪个类别最多（很有可能全部是同一个label）

        # 找出最大选票
        mmax = max(class_count)

        # 找出最大选票标签
        # 可以使用argmax ? 找最大值的下标
        for i in range(class_total):
            if mmax == class_count[i]:
                predict.append(i)
                break

    return np.array(predict)


k = 10

if __name__ == '__main__':
    print('Start read data')
    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv',header=0) # header=0,第0行是index
    data = raw_data.values
    print("wtx data.shape = ", data.shape)

    imgs = data[0::,1::] # 切片：行（第0维）：全部获取； 列（第1维）：从第1列开始获取（因为第0列是value）
    labels = data[::,0]  # 切片：取第0列
    print("wtx, imgs.shape = ", imgs.shape)
    print("wtx, labels.shape = ", labels.shape)

    # 对所有的数据先提取特征，然后再划分训练集和测试集
    features = get_hog_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print('read data cost ',time_2 - time_1,' second','\n')

    print('Start training')
    print('knn do not need to train')
    time_3 = time.time()
    print('training cost ',time_3 - time_2,' second','\n')

    print('Start predicting')
    test_predict = Predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print("The accruacy socre is ", score)