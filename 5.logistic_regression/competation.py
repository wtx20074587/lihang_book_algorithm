# encoding=utf-8
# @Author: WenDesi
# @Date:   08-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 08-11-16

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logistic_regression import *
from binary_perceptron import  *

if __name__ == '__main__':

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    test_time = 10

    p = Perceptron()
    lr = LogisticRegression()
    writer = csv.writer(open('result.csv', 'wb'))

    for time in range(test_time):
        print('iterater time %d' % time)

        train_features, test_features, train_labels, test_labels = train_test_split(
            imgs, labels, test_size=0.33, random_state=23323)

        p.train(train_features, train_labels)
        lr.train(train_features, train_labels)

        p_predict = p.predict(test_features)
        lr_predict = lr.predict(test_features)

        p_score = accuracy_score(test_labels, p_predict)
        lr_score = accuracy_score(test_labels, lr_predict)

        print('perceptron accruacy score ', p_score)
        print('logistic Regression accruacy score ', lr_score)

        writer.writerow([time,p_score,lr_score])
