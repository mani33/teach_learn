# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 16:08:27 2026
Problems for Anirudh
@author: Mani Subramaniyan
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

fn = 'C:/Users/maniv/Downloads/girl.jpg'

imc = cv2.imread(fn)
imc = cv2.cvtColor(imc, cv2.COLOR_BGR2RGB)
im = cv2.cvtColor(imc, cv2.COLOR_RGB2GRAY)

ker_x = np.array([[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]])

ker_y = ker_x.copy().T

imx = sig.convolve2d(im, ker_x)
imy = sig.convolve2d(im, ker_y)

eim = np.sqrt(imx**2 + imy**2)
eim = 255 * eim/np.max(eim)

eim[eim < 20] = 0
plt.subplot(1,4,1)
plt.imshow(imc)

plt.subplot(1,4,2)
plt.imshow(imx.astype(np.uint8), cmap='gray')

plt.subplot(1,4,3)
plt.imshow(imy.astype(np.uint8), cmap='gray')

plt.subplot(1,4,4)
plt.imshow(eim.astype(np.uint8), cmap='gray')

