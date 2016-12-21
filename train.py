#!/usr/bin/env python

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    if len(sys.argv) != 2:
        print 'Usage: python train.py <data_folder>'

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

    # prepare training data
    sys.stdout.write("Preparing data ")
    sys.stdout.flush()
    i = start
    road_masks = []

    while(i <= end):

        if i % 10 == 0:
            sys.stdout.write('. ')
            sys.stdout.flush()

        #joystickVals[i-start,:]

        flag, frame = vid.read()

        frame = cv2.resize(frame, (160,120))

        road_masks.append( frame.flatten().astype(np.float32) )

        i += 1

    joystickVals = np.loadtxt(dataDir+'joystick.csv', delimiter=',')[start:end+1,1:]

    print ' Done!'

    # train
    sys.stdout.write('Training AI . . . ')
    sys.stdout.flush()
    ai = cv2.ANN_MLP()

    samples = np.asarray(road_masks)

    sample_n, var_n = samples.shape
    sample_n, out_n = joystickVals.shape

    layer_sizes = np.int32([var_n, 1000, out_n])
    ai.create(layer_sizes)

    # CvANN_MLP_TrainParams::BACKPROP,0.001
    params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 300, 0.01),
                   train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                   bp_dw_scale = 0.001,
                   bp_moment_scale = 0.0 )

    ai.train(samples, joystickVals, None, params = params)
    ai.save("ai.xml")
    print ' Done!'

    return

if __name__ == '__main__':
    main()
