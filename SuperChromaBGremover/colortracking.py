import cv2
import numpy as np
import pygame

iy,ix=-1,-1
image_mask=None
def pick_color(event,x,y,flags,param):
    global ix,iy,image_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy=x,y
        image_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', pick_color)
        cv2.imshow("Frame",image_hsv)
        pixel = image_hsv[iy,ix]
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        image_mask += cv2.inRange(image_hsv,lower,upper)
       
       
       
if __name__=="__main__":
    global image_hsv, pixel
    
    image_hsv = None   # global ;(
    pixel = (20,60,80)
    cap = cv2.VideoCapture(0)
    ret,frame=cap.read()
    subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    kernel = np.ones((3,3), np.uint8)
    bg_img=cv2.imread('beach.jpg')
    masks=[]

    upper1 =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
    lower1 =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
    image_mask = cv2.inRange(frame,lower1,upper1)
    
    # pygame.init()
    while True:
        
        _, frame = cap.read()        
        prev_frame=frame
        # frame=frame[:300,:500,:]
        # bg_cropped=bg_img[:300,:500,:]
        
        
        # lower=np.array([0,20,70], dtype=np.uint8)
        # upper=np.array([20,255,255], dtype=np.uint8)
        # mouse_pos=pygame.mouse.get_pos()
        # print(mouse_pos)
        cv2.namedWindow("Mask")
        cv2.setMouseCallback("Mask", pick_color)
        
        #mask=tracker.image_mask\
        masked_img=np.copy(frame)
        masked_img[image_mask!=0]=[0,0,0]
        # masked_img=masked_img+bg_cropped
        cv2.imshow("Mask",masked_img)
        key = cv2.waitKey(100)
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()    
