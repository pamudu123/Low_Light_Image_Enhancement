import cv2
import cvzone
import numpy as np
from Noise_Adder import noisy
#%%
# gauss
# s&p
# poisson
# speckle


img = cv2.imread('samples/Img_1.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img2[:,:,0] = np.clip(img2[:,:,0] - 50, a_min = 0, a_max = 255)
# img2[:,:,1] = np.clip(img2[:,:,1] - 50, a_min = 0, a_max = 255)
# img2[:,:,2] = np.clip(img2[:,:,2] - 50, a_min = 0, a_max = 255)
img_black = np.zeros((img.shape[0], img.shape[1], 3), dtype = "uint8")

_,img_thresh1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
_,img_thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)


#th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

dst_img1 = cv2.addWeighted(img,0.6,img_thresh1,0.4,0)
dst_img2 = cv2.addWeighted(img,0.6,img_thresh2,0.4,0)
dst_img3 = cv2.addWeighted(img,0.2,img_black,4,0)
#dst_img4 = cv2.addWeighted(img2,0.5,th2,0.5,0)

#sharp Image
gauss = cv2.GaussianBlur(img, (7,7), 0)
unsharp_image = cv2.addWeighted(img, 2, gauss, -1, 0)
dst_img4 = cv2.addWeighted(unsharp_image,0.2,img_black,4,0)
n_dst_img4 = noisy("s&p",dst_img4)

#%%
cv2.imwrite('save_images/img.png', img)
cv2.imwrite('save_images/img_b.png', dst_img3)
cv2.imwrite('save_images/img_bg.png', dst_img4)
cv2.imwrite('save_images/img_bgn.png', n_dst_img4)


imgList = [dst_img3,dst_img4,n_dst_img4]
stackedImg = cvzone.stackImages(imgList,2,0.35)
cv2.imshow("LIVE", stackedImg)
key = cv2.waitKey(0)
if key == ord('q'):
    cv2.destroyAllWindows()
