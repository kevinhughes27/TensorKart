#!/usr/bin/env python

import sys
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

        # load sample
        image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
        joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5))

        # add joystick values to y
        y.append(joystick_values)

        # load, prepare and add images to X
        for image_file in image_files:
            image = imread(image_file)
            vec = prepare_image(image)
            X.append(vec)

    print "Saving to file..."
    X = np.asarray(X)
    y = np.concatenate(y)

    np.save("data/X", X)
    np.save("data/y", y)

    print "Done!"
    return


if __name__ == '__main__':
    main()
