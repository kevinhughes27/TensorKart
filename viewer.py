#!/usr/bin/env python

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    if len(sys.argv) != 2:
        print 'Usage: python viewer.py <data_folder>'

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
    plotData = []
    plt.ion()
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    while(i <= end):

        # joystick
        print i, " ", joystickVals[i-start,:]

        # plot
        plotData.append( joystickVals[i-start,:] )
        if len(plotData) > 30:
            plotData.pop(0)
        x = np.asarray(plotData)

        plt.plot(range(i,i+len(plotData)), x[:,0], 'r')
        plt.hold(True)
        plt.plot(range(i,i+len(plotData)), x[:,1], 'b')
        plt.plot(range(i,i+len(plotData)), x[:,2], 'g')
        plt.plot(range(i,i+len(plotData)), x[:,3], 'k')
        plt.plot(range(i,i+len(plotData)), x[:,4], 'y')
        plt.draw()
        plt.hold(False)

        #image
        flag, frame = vid.read()
        cv2.imshow("frame", frame)

        cv2.waitKey(30)

        i += 1

    return

if __name__ == '__main__':
    main()
