import numpy as np

# 1.数据集（扩充perceptron中的数据）,6个正例，3个负例
data_coord = np.asarray(((3, 3), (4, 3), (5, 5), (4, 5), (5, 4), (3, 5),
                         (1, 1), (0, 0), (1, -1)))
data_label = np.asarray((1, 1, 1, 1, 1, 1,
                         -1, -1, -1))

# 2.测试数据
test_data = np.asarray((0.3, 0.4))
K = 2


# 计算两点的距离（两个向量）
def get_dis(a, b):
    return np.linalg.norm(a - b)


# knn
def knn(test_data, train_data, train_label, K):
    max_distance = np.Inf
    # test_data k个邻居的距离
    knn_list = list((max_distance - i) for i in range(K))
    # test_data k个邻居,的下标
    label_list = list(-1 for i in range(K))
    label = 0

    for i in range(len(train_label)):
        vec_train = train_data[i]
        label_train = train_label[i]
        # 计算train集合中，每个点与test_data的距离
        test_train_dist = get_dis(test_data, vec_train)
        curr_max_knn = np.argmax(knn_list)
        curr_max_dist = knn_list[curr_max_knn]
        # 找到train集合中，跟test_data距离最近的K个点
        if test_train_dist < curr_max_dist:
            knn_list[curr_max_knn] = curr_max_dist
            label_list[curr_max_knn] = label_train

    # 在knn_list中，“投票表决”
    print("label_list = ", label_list)
    outcome = np.sum(label_list)
    if outcome > 0:
        label = 1
    elif outcome < 0:
        label = -1
    print("label = ", label)
    return label;


# run it
knn(test_data=test_data, train_data=data_coord, train_label=data_label, K=K)
