# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from numpy.matlib import repmat
import pickle
from os.path import join
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def cable_length(station_order, x, y, cx, cy):
    length = 0
    for i in range(cx.size):
        for j in range(6):
            station_idx = station_order[i * 6 + j]
            dx = cx[i] - x[station_idx]
            dy = cy[i] - y[station_idx]
            dr = (dx**2 + dy**2)**0.5
            length += dr
    return length

# def callback(xk, convergence=val):
#     print(xk)


def main():
    filename = join('scripts', 'TEMP_results', 'model02_r08_layout.p')
    layout = pickle.load(open(filename, 'rb'))
    print(layout.keys())
    x = layout['x']
    y = layout['y']
    cx = layout['cx']
    cy = layout['cy']
    num_stations = x.size
    x0 = np.arange(x.size, dtype=np.int)

    print(cable_length(x0, x, y, cx, cy) / 1e3)
    bounds = repmat((0, num_stations - 1), num_stations, 1)

    result = differential_evolution(cable_length, bounds, (x, y, cx, cy),
                                    maxiter=5, disp=True)
    print(result)


    # fig, ax = plt.subplots(figsize=(8, 8))
    # plt.plot(x, y, 'k.')
    # plt.plot(cx, cy, 'r+')
    # plt.show()


if __name__ == '__main__':
    main()



