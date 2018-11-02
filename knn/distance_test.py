#encoding=utf-8

import pandas as pd
import numpy as np
import time

if __name__ == '__main__':
    vec_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    vec_2 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,]

    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    time_1 = time.time()
    # 使用数学方法，求解距离问题
    print(np.sqrt(np.sum(np.square(vec_1 - vec_2))))

    time_2 = time.time()
    print(time_2-time_1)
    # 1.linalg : lin(line  线性), alg(algebra 代数), nrom: 范数
    # param: ord=1:1范数， ord=2:2范数, 默认2范数
    print(np.linalg.norm(vec_1 - vec_2,ord=2))
    time_3 = time.time()
    print(time_3-time_2)
