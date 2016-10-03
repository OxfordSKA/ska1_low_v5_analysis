# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from math import log10, log


def lin_intep_1(x, x0=500, x1=6400, y0=50, y1=200):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def lin_intep_2(x, x0=500, x1=6400, y0=50, y1=200):
    a = x1 - x
    b = x - x0
    f = b / (a + b)
    return f * y1 + (1 - f) * y0

def log_intep(x, x0=500, x1=6400, y0=50, y1=200):
    a = x1 - x
    b = x - x0
    f = b / (a + b)
    return y1**f * y0**(1 - f)

if __name__ == '__main__':
    x0, x1 = 500, 6400
    y0, y1 = 300, 1213.2765
    fig, ax = plt.subplots(figsize=(8, 8))
    for x in np.linspace(x0, x1, 50):
        y = lin_intep_1(x, x0, x1, y0, y1)
        ax.plot([x], [y], 'k+')
        y = lin_intep_2(x, x0, x1, y0, y1)
        ax.plot([x], [y], 'rx')
        y = log_intep(x, x0, x1, y0, y1)
        ax.plot([x], [y], 'go')
    plt.show()
