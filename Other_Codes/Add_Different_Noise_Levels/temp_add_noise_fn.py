import cv2
import cvzone
import numpy as np
from Noise_Adder import noisy
import os


def temp_add_noise_fn(img,img_n,display=False,save=True):
    #img = cv2.imread(img_path)
    #img_name =  img_path.split('/')[-1].split('.')[0]
    
    img_black = np.zeros((img.shape[0], img.shape[1], 3), dtype = "uint8")
    
    #gauss = cv2.GaussianBlur(img, (7,7), 0)
    #unsharp_image = cv2.addWeighted(img, 2, gauss, -1, 0)
    
    # dst_img_b = cv2.addWeighted(img,0.05,img_black,10,0)
    dst_img_bg = cv2.addWeighted(img,0.05, img_black,1, 0)
    # dst_img_bgn = noisy("s&p",dst_img_bg)
    
    if save:
        save_path = r'save_images/Folder_2'
        # cv2.imwrite(f'{save_path}/img_{img_n}_n2.png', img)
        # cv2.imwrite(f'save_images/img_{img_n}_b_L1.png', dst_img_b)
        #cv2.imwrite(f'save_images/img_{img_n}_bg_L8.png', dst_img_bg)
        cv2.imwrite(f'save_images/img_org.png', img)
    # cv2.imwrite(f'{save_path}/img_{img_n}_bgn.png', dst_img_bgn)
    
    if display:
        imgList = [img, dst_img_bg]
        stackedImg = cvzone.stackImages(imgList,2,0.5)
        cv2.imshow("LIVE", stackedImg)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()


# Test the function
# img_folder = r'C:\Users\hp\Desktop\FYP\Light_Primash_Image\part_1'
#
# display = True
#
# img_names = os.listdir(img_folder)
#
# for img_name in img_names:
#     img_path = os.path.join(img_folder,img_name)
#     img = cv2.imread(img_path)
#
#     temp_add_noise_fn(img,10,display=True,save=False)