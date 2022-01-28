from PIL import Image
import numpy as np
import cv2
def kmeans(imga,num,cntt=100):
    np.random.rand(0)
    img=imga.reshape(-1,3)
    cnt=0
    value=np.array(np.random.rand(num,3)*255,dtype=float)
    while True:
        cnt+=1
        flag=np.copy(value)
        mid=[]
        for i in range(num):
            mid.append(np.square(img-value[i]))
        mid=np.sum(mid,axis=2)
        label=np.argmin(mid,axis=0)
        for j in range(num):
            value[j]=img[label==j].mean(axis=0)
        if np.sum(np.abs(value-flag))<1e-2 or cnt==cntt:
            break
    graph=np.array(value[label].reshape(imga.shape),dtype=np.uint8)#此处应将value[label]整体看作一个矩阵来理解
    return graph
imga=cv2.imread("G:\\11.jpg")
#imga=np.array(Image.open("G:\\11.jpg"))#第二种读取方式
graph=kmeans(imga,num=5)
cv2.imshow("11",graph)#注意：此处一定要有图片名，且要与上一行的图片名一致
cv2.waitKey(0)