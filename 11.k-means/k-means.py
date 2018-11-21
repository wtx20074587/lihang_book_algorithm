# coding=utf-8
import numpy as np

# 计算两个向量的距离，用的是欧几里得距离
def get_dis(vecA, vecB):
    return np.linalg.norm(vecA - vecB)

# 随机生成初始的质心（ng的课说的初始方式是随机选K个点）
def randCent(dataSet, k):
    p = np.shape(dataSet)[1] # 维度p
    centroids = np.mat(np.zeros((k, p)))
    # k个簇心
    for j in range(p):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(np.array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k):
    n = np.shape(dataSet)[0] # N个
    p = np.shape(dataSet)[1]
    clusterAssment = np.mat(np.zeros((n,p))) # 记录每个点的簇心
    # 随机生成簇心
    centroids = randCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(n):  # 计算每个点簇心
            minDist = np.inf
            minIndex = -1
            for j in range(k): # 当前点跟k个簇心的距离,选最近的簇心为自己的簇心
                distJI = get_dis(centroids[j, :], dataSet[i, :])
                print("distJI = ",distJI)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True # 循环标记
            clusterAssment[i, :] = (minIndex, minDist)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # matrix.A 将matrix转换为array
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def main():
    # 1.随机生成几组数据（K = 2）
    data_coord = np.asarray(((3, 3), (4, 3), (5, 5), (4, 5), (5, 4), (3, 5), (21, 22), (18, 23), (20, 23)))
    dataMat = np.mat(data_coord)
    myCentroids, clustAssing = kMeans(dataMat, 2)
    print(myCentroids)

if __name__ == '__main__':
    main()
