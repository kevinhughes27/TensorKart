#!/usr/bin/env python

import sys
import json
import numpy as np
from skimage.io import imread
from utils import prepare_image


def main():
    if len(sys.argv) <= 2:
        print 'Usage: python prepare.py [data_folders]'
        return

    samples = sys.argv[1:]

    # prepare training data
    print "Preparing data"

    X = []
    y = []

    for sample in samples:
        print sample
        if sample[-1] != '/':
            sample+='/'

        # load dataset info
        with open(sample+'info.json') as data_file:
            data = json.load(data_file)
            start = data['start']
            end = data['end']

        # load, reshape and add images to X
        for i in range(start, end):
            image_file = sample+"img_%i.png" % (i)
            image = imread(image_file)
            vec = prepare_image(image)
            X.append(vec)

        # add joystick values to y
        joystick_values = np.loadtxt(sample+'joystick.csv', delimiter=',')[start:end+1,1:]
        y.append(joystick_values)

    print "Saving to file..."
    X = np.asarray(X)
    y = np.concatenate(y)

    np.save("data/X", X)
    np.save("data/y", y)

    print "Done!"
    return

if __name__ == '__main__':
    main()
