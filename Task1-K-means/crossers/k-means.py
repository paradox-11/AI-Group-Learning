import numpy as np
def kmeans(image,k,max=100000,seed=0):
    img = image.reshape(-1, 3)
    index=0
    clus_value=np.array(np.random.rand(3,k)*255, dtype=float)
    while True:
        index+=1
        flag=np.copy(clus_value)
        cs=np.array([np.square(img-clus_value[0]), np.square(img-clus_value[1]),np.square(img-clus_value[2])])
        cs=np.sum(cs, axis=2)
        label=np.argmin(cs, axis=0)
        for j in range(k):
            clus_value[j]=img[label==j].mean(axis=0)
        if np.sum(np.abs(clus_value-flag))<1e-8 or index==max:
            break
    new_image=np.array(clus_value[label].reshape(image.shape),dtype=np.uint8)
    return new_image


import cv2 as cv
img=cv.imread("C:\\Users\\86182\\Downloads\\d9e8a4ced6b21804f09f508a624497b9.jpeg")
new_image=kmeans(img, 3, 100000, 0)
cv.imshow("image", new_image)
cv.waitKey(0)