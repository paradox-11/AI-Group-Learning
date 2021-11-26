import cv2
import numpy as np

img = cv2.imread("K-means/test.jpg",cv2.IMREAD_COLOR)

def kmeans(image,k,max_iter=10000):
    img = image.reshape(-1,3)
    index = 0  #迭代次数

    center_value = np.array(np.random.rand(k,3)*255,dtype=np.double) #随机设置中心点
    while True:
        index += 1
        flag = center_value.copy()

        '''计算样本点与中心的距离'''
        distance = np.array([np.square(img-flag[0])])
        for i in range(1,k):
            distance = np.append(distance,np.array([np.square(img-flag[i])]),axis=0)
        distance = np.sum(distance,axis=2)

        labels = np.argmin(distance,axis=0)  #选出每个样本点的最近中心点
        
        '''重新计算中心点'''
        for j in range(k):
            center_value[j] = img[labels==j].mean(axis=0)
        if np.sum(np.abs(center_value-flag)) < 1e-8 or index == max_iter:
            break

    img = np.array(center_value[labels].reshape(image.shape),dtype=np.uint8)
    return img


img_result = kmeans(img,3)
cv2.imwrite("K-means/result.jpg", img_result)
