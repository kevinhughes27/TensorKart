#!/usr/bin/env python

import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    if len(sys.argv) != 2:
        print 'Usage: python viewer.py <data_folder>'
        return

    dataDir = sys.argv[1]
    if dataDir[-1] != '/':
        dataDir += '/'

    # load dataset info
    with open(dataDir+'info.json') as data_file:
        data = json.load(data_file)
        start = data['start']
        end = data['end']

    joystickVals = np.loadtxt(dataDir+'joystick.csv', delimiter=',')[start:-1,1:]
    plotData = []

    plt.ion()
    plt.figure('viewer', figsize=(16, 6))

    i = start
    while(i <= end):

        # joystick
        print i, " ", joystickVals[i-start,:]

        # format data
        plotData.append( joystickVals[i-start,:] )
        if len(plotData) > 30:
            plotData.pop(0)
        x = np.asarray(plotData)

        # image
        plt.subplot(121)
        image_file = dataDir+"img_%i.png" % (i)
        img = mpimg.imread(image_file)
        plt.imshow(img)

        # plot
        plt.subplot(122)
        plt.plot(range(i,i+len(plotData)), x[:,0], 'r')
        plt.hold(True)
        plt.plot(range(i,i+len(plotData)), x[:,1], 'b')
        plt.plot(range(i,i+len(plotData)), x[:,2], 'g')
        plt.plot(range(i,i+len(plotData)), x[:,3], 'k')
        plt.plot(range(i,i+len(plotData)), x[:,4], 'y')
        plt.draw()
        plt.hold(False)

        plt.pause(0.5)
        i += 1

    return

if __name__ == '__main__':
    main()
