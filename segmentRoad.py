import cv2
import numpy as np

def segmentRoad(frame):
    # raw frame
    #cv2.imshow("frame", frame)
    
    # edge detection
    #edges = cv2.Canny(frame,50,100)
    #cv2.imshow("edges", edges)
    
    # flood fill
    flooded = frame.copy()
    w,h = frame.shape[0:2]
    mask = np.zeros([w+2,h+2], np.uint8)
    seed_pt = (h/2,w-40)
    lo = 3
    hi = 3
    cv2.floodFill(flooded, mask, seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3)
    #cv2.circle(flooded, seed_pt, 2, (0, 0, 255), -1)
    #cv2.imshow("flooded", flooded)
    mask = cv2.normalize(mask,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # connectected components
    #contours, _ = cv2.findContours(mask, cv2.cv.CV_RETR_LIST, cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    #mask = np.zeros([w+2,h+2,3], np.uint8)
    #biggest_c = contours[0]
    #for c in contours:
        #if cv2.contourArea(c) > cv2.contourArea(biggest_c):
            #biggest_c = c    
	#cv2.drawContours(mask,[biggest_c],-1,255,-1)
    
    # approx poly
    #curve = cv2.approxPolyDP(biggest_c, 0.5, False)
    #for pt in curve:
        #cv2.circle(mask, (pt[:,0],pt[:,1]), 2, (0, 0, 255), -1)
    
    # sobel
    #mask = cv2.Sobel(mask, cv2.cv.CV_8U, 1, 1, 7)
    #cv2.imshow("mask", mask)
    #cv2.waitKey(100)
    
    return mask
