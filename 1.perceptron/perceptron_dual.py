import numpy as np

def perceptron_dual(x_input,y_input,gram):
    alpha_star = np.zeros(y_input.shape[0]) # alpha矩阵,x_input为n行矩阵
    b_star     = 0
    learning_rate = 1
    classification_right_count = 0 # 正确分类计数
    while True:
        for i in range(x_input.shape[0]):
            y = y_input[i]
            # 判断是否满足条件
            value = (np.sum(gram[i]*(alpha_star*y_input)) + b_star)
            if y*value <= 0:
                alpha_star[i] += learning_rate
                b_star += y
                print("update , alpha =", alpha_star)
                print("update , b_star = ",b_star)
            else:
                classification_right_count += 1
        # 若都已经分类正确，则退出
        if classification_right_count >= y_input.shape[0]: # y_input,行
            print("end, alpha = ",alpha_star)
            print("end, b_star = " , b_star)
            break
        # 否则，继续循环
        else:
            classification_right_count = 0

# 1.准备数据
data_coord = np.asarray(((3, 3), (4, 3), (1, 1)))
data_label = np.asarray((1, 1, -1))
# 2.计算gram矩阵
x = np.asarray([data_coord[0],data_coord[1],data_coord[2]])
gram = np.matmul(x,x.T)
print("gram = ",gram)
# 3.感知机 对偶形式求解
perceptron_dual(data_coord,data_label,gram)