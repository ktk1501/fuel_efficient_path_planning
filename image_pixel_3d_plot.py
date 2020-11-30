import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

img = cv2.imread('terrain2.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("asdf", img_gray)
cv2.waitKey(2)
#plt.imshow(img_gray, cmap='gray')

xx, yy = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
fig = plt.figure(figsize=(15, 15))
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, img_gray, rstride=4, cstride=4,
                cmap=plt.cm.gray, linewidth=2, alpha=0.6)
ax.set_zlim3d(0,256)
tx = [2,3,4,5]
ty = [5,6,7,8]
height = [img_gray[i][j]+3 for (i,j) in zip(tx,ty)]
ax.plot(tx, ty, height, alpha=1, marker='v', c='r')
plt.plot(tx, ty, "*b")
#ax.view_init(80, 30)

plt.show()
