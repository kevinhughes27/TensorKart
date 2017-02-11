#!/usr/bin/env python

import sys
import time
import array

import wx
wx.App()

from skimage.transform import resize
from skimage.util import img_as_float

import numpy as np

from PIL import Image


# CONSTANTS #
SRC_W = 640
SRC_H = 480
SRC_D = 3
OFFSET_X = 400
OFFSET_Y = 240

IMG_W = 200
IMG_H = 66
IMG_D = 3


# ORIGINAL take_screenshot() #
def take_screenshot():
    screen = wx.ScreenDC()
    size = screen.GetSize()
    bmp = wx.Bitmap(size[0], size[1])
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
    return bmp.GetSubBitmap(wx.Rect([0,0],[SRC_W,SRC_H]))


# MODIFIED take_screenshot() #
def take_screenshot_direct_screen_rect():
    screen = wx.ScreenDC()
    bmp = wx.Bitmap(SRC_W, SRC_H)
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, SRC_W, SRC_H, screen, OFFSET_X, OFFSET_Y)
    return bmp


# ORIGINAL prepare_image() #
def prepare_image(img):
    buf = img.ConvertToImage().GetData()
    img = np.frombuffer(buf, dtype='uint8')

    img = img.reshape(SRC_H, SRC_W, SRC_D)
    img = resize(img, [IMG_H, IMG_W])

    return img


# MODIFIED prepare_image() #
arr = array.array('B', [0] * (SRC_W * SRC_H * SRC_D));
def prepare_image_with_PIL(img):
    img.CopyToBuffer(arr)
    img = np.frombuffer(arr, dtype=np.uint8)

    img = img.reshape(SRC_H, SRC_W, SRC_D)

    im = Image.fromarray(img)
    im = im.resize((IMG_W, IMG_H))

    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((IMG_H, IMG_W, IMG_D))

    return img_as_float(im_arr)


# CALL ORIGINAL #
def call_original():
    bmp = take_screenshot()
    vec = prepare_image(bmp)


# CALL MODIFIED #
def call_modified():
    bmp = take_screenshot_direct_screen_rect()
    vec = prepare_image_with_PIL(bmp)


# MAIN #
if __name__ == '__main__':
  import timeit

  try:
    n = int(sys.argv[1])
  except (ValueError, IndexError) as e:
    n = 100

  print("# Running tests " + str(n) + " times")

  print("#")
  print("# ORIGINAL:")
  print(timeit.timeit("call_original()", setup="from __main__ import call_original;", number=n))

  print("#")
  print("# MODIFIED:")
  print(timeit.timeit("call_modified()", setup="from __main__ import call_modified;", number=n))
 

######################################################
# SOME RESULTS #
#
# Running tests 10000 times
#
# ORIGINAL:
# 1210.20094013
#
# MODIFIED:
# 313.987584114
#
#
# Running tests 10000 times
#
# ORIGINAL:
# 1074.97350001
#
# MODIFIED:
# 270.604922056
######################################################
  

######################################################
# RESULTS DURING ACTUAL PLAY.PY RUN #
#
# ORIGINAL CODE:
#                  Screenshot        Prepare Image   Model Eval
# Avg Times (500): 0.000318816661835 0.136291568279  0.0443236446381
#
# MODIFIED CODE:
#                  Screenshot        Prepare Image   Model Eval
# Avg Times (500): 0.000203844547272 0.0492500219345 0.0412494616508
#
# IMPROVEMENT (AS % DECREASE OVER ORIGINAL):
#                  Screenshot        Prepare Image   Model Eval
#                  36.06%            63.86%          6.94%
#
######################################################


