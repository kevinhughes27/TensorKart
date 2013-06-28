#!/usr/bin/env python

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segmentRoad import segmentRoad

def main():

    if len(sys.argv) != 2:
        print 'Usage: python trainAI.py <data_folder>'

    dataDir = sys.argv[1]
    
    if dataDir[-1] != '/':
        dataDir+='/'
    
    info = open(dataDir+'info.txt')
    info.readline()
    start = int(info.readline())
    end = int(info.readline())
    
    vid = cv2.VideoCapture()
    vid.open(dataDir+'img_%0d.png')
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
    
    joystickVals = np.loadtxt(dataDir+'joystick.csv', delimiter=',')[start:-1,1:]
    
    i = start
    while(i <= end):
        
        flag, frame = vid.read()
        
        segmentRoad(frame)
        
        #joystickVals[i-start,:]
        
        i += 1
        
    return       
        
if __name__ == '__main__':
    main()
