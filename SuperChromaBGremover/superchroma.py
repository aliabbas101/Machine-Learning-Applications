import cv2
import numpy as np 
from PIL import ImageGrab



cap=cv2.VideoCapture(0)

# cap=cv2.VideoCapture(0)
panel=np.zeros([100,700], np.uint8)
cv2.namedWindow("panel")

def nothing(x):
    pass
cv2.createTrackbar("L- h","panel",0,255,nothing)
cv2.createTrackbar("U- h","panel",255,255,nothing)

cv2.createTrackbar("L- s","panel",0,255,nothing)
cv2.createTrackbar("U- s","panel",255,255,nothing)

cv2.createTrackbar("L- v","panel",0,255,nothing)
cv2.createTrackbar("U- v","panel",255,255,nothing)


bg_org1=ImageGrab.grab()
bg_org1 = cv2.cvtColor(np.asarray(bg_org1), cv2.COLOR_RGB2BGR)
r = cv2.selectROI(bg_org1)
disparea=bg_org1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
scale_percent = 60 # percent of original size
width = int(disparea.shape[1])
height = int(disparea.shape[0])
dim = (width, height)

while True:
    _,frame=cap.read()
    bg_img=ImageGrab.grab()
    bg_org=ImageGrab.grab()
    
    bg_img = cv2.cvtColor(np.asarray(bg_img), cv2.COLOR_RGB2BGR)
    bg_org = cv2.cvtColor(np.asarray(bg_org), cv2.COLOR_RGB2BGR)
    disparea=bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    
    frame=frame[:480,:640,:]
    bg_cropped=bg_img[:480,:640,:]
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h=cv2.getTrackbarPos("L- h","panel")
    u_h=cv2.getTrackbarPos("U- h","panel")
    
    l_s=cv2.getTrackbarPos("L- s","panel")
    u_s=cv2.getTrackbarPos("U- s","panel")
    
    l_v=cv2.getTrackbarPos("L- v","panel")
    u_v=cv2.getTrackbarPos("U- v","panel")
    lower_green=np.array([l_h,l_s,l_v])
    upper_green=np.array([u_h,u_s,u_v])
    # lower_green=np.array([73,-5,149])
    # upper_green=np.array([93,15,229])
    mask=cv2.inRange(hsv,lower_green, upper_green)
    mask_inv=cv2.bitwise_not(mask)
    
    bg=cv2.bitwise_and(frame,frame, mask=mask)
    fg=cv2.bitwise_and(frame,frame, mask=mask_inv)
    resized_fg=cv2.resize(fg,dim, interpolation=cv2.INTER_AREA)
    bg_cropped[mask_inv!=0]=[0,0,0]
    
    
    hsv_disparea=cv2.cvtColor(disparea, cv2.COLOR_BGR2HSV)
    mask_disparea=cv2.inRange(hsv_disparea,lower_green, upper_green)
    mask_inv_disparea=cv2.bitwise_not(mask_disparea)
    bg_resized=cv2.bitwise_and(bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])],bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])],mask=mask_disparea)
    fg_resized=cv2.bitwise_and(bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])],bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])],mask=mask_inv_disparea)
    disparea=bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    disparea[mask_inv_disparea!=0]=[0,0,0]
    
    bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=[0,0,0]
    bg_org[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=disparea+resized_fg
    cv2.imshow("panel",panel)
    cv2.imshow("bg",bg)
    cv2.imshow("bgorg",bg_org)
    
    cv2.imshow("bgc",bg_cropped+fg)
    cv2.imshow("fg",fg)
    cv2.imshow("disparea",disparea+resized_fg)
    k=cv2.waitKey(30) & 0xFF
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()