import cv2
import cvzone
import numpy as np
import os
from temp_add_noise_fn import temp_add_noise_fn

#%%
video_path = r'samples/signrecognition.mp4'
cap = cv2.VideoCapture(video_path)

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frames_needed = 1
skip_freq = int(total_frames/frames_needed)

c = 0
i = 0 
while True:
    ret, frame = cap.read()
    if not ret: break
    
    if c % skip_freq == 0:
        temp_add_noise_fn(frame, i, True)
        print(i)
        i += 1
          
    c += 1
    #cv2.imshow("LIVE_VIDEO",frame) 
    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
        
cap.release()
cv2.destroyAllWindows()

