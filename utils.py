import pygame

import wx
wx.App()

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

IMG_W = 200
IMG_H = 66


def take_screenshot():
    screen = wx.ScreenDC()
    size = screen.GetSize()
    bmp = wx.Bitmap(size[0], size[1])
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
    return bmp.GetSubBitmap(wx.Rect([0,0],[640,480]))


def prepare_image(img):
    if(type(img) == wx._core.Bitmap):
        buf = img.ConvertToImage().GetData()
        img = np.frombuffer(buf, dtype='uint8')

    img = img.reshape(480, 640, 3)
    img = resize(img, [IMG_H, IMG_W])

    return img


class XboxController:
    def __init__(self):
        try:
            pygame.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except:
            print 'unable to connect to Xbox Controller'


    def read(self):
        pygame.event.pump()
        x = self.joystick.get_axis(0)
        y = self.joystick.get_axis(1)
        a = self.joystick.get_button(0)
        b = self.joystick.get_button(2) # b=1, x=2
        rb = self.joystick.get_button(5)
        return [x, y, a, b, rb]


    # used to toggle manual override for now
    def btn_b(self):
        pygame.event.pump()
        return self.joystick.get_button(1)


class Data(object):
    def __init__(self):
        self._X = np.load("data/X.npy")
        self._y = np.load("data/y.npy")
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._X.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
