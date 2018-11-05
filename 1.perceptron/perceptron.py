import numpy as np

# 计算y值
def cacl_y(w, x, b):
    # print(w)
    # print(w.shape)
    # print(x)
    # print(x.shape)
    return np.sign(np.matmul(np.transpose(w), x) + b)

# 感知机计算过程
def perceptron(data_coord, data_label):
    # 0. 初始化参数:w,b, learning_rate
    learning_rate = 1
    w_star = np.zeros(shape=data_coord[0].shape) #zeros
    b_star = 0

    #1.开启更新w，b的循环:
    # 假设没有不可分的数据，当全部数据分类正确时，停止循环
    while True:
        count = 0;
        for i in range(len(data_coord)):
            # 2.1 对每个数据，查看分类是否错误
            x = data_coord[i]
            y_ = data_label[i]
            y = cacl_y(w_star, x, b_star)
            print("y_ = ", y_)
            print("y = ", y)
            print("\n\n")
            # 2.2 若分类错误(不大于0)，更新w，b
            if y * y_ <= 0:  # update w,b
                w_star += learning_rate * y_ * x
                b_star += learning_rate * y_
                print("w_star = ",w_star)
                print("b_star = ",b_star)

            # 2.2 分类正确，对分类正确的数据个数 计数
            else:
                print("count = ", count)
                count += 1
            print("One Circle Again!")
        print("After going through all data, count = ", count)

        # 3.1 对所有数据分类正确，stop circle
        if count == len(data_label):
            print("\n\n")
            break;
        # 3.2 否则，重启 遍历数据过程
        else:
            count = 0;
    # 4.结束：输出 w b
    print("w_star = ", w_star)
    print("b_star = ", b_star)

# 准备3组测试数据，2个正例， 1个负例
data_coord = np.asarray(((3, 3), (4, 3), (1, 1)))
data_label = np.asarray((1, 1, -1))
# startperceptron
perceptron(data_coord, data_label)
