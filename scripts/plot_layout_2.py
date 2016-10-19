# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle
from os.path import join

data = np.loadtxt(join(b'v7_coords', b'ska1_low_v7_enu.txt'))
x = data[:, 0]
y = data[:, 1]

num_core = 224
num_rings = 69
num_spiral = 93
num_outer = 512 - num_core - num_rings - num_spiral

r = x**2 + y**2
sort_idx = np.argsort(r)
x = x[sort_idx]
y = y[sort_idx]

x1 = x[:num_core]
y1 = y[:num_core]
x2 = x[num_core:num_core + num_rings]
y2 = y[num_core:num_core + num_rings]
x3 = x[num_core + num_rings: num_core + num_rings + num_spiral]
y3 = y[num_core + num_rings: num_core + num_rings + num_spiral]
x4 = x[num_core + num_rings + num_spiral:]
y4 = y[num_core + num_rings + num_spiral:]

print(x1.size)
print(x2.size)
print(x3.size)
print(x4.size)


def _create_figure():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.04, top=1.0)
    return fig, ax


def plot1():
    fig, ax = _create_figure()

    for i, xy in enumerate(zip(x1, y1)):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=False, color='b', lw=1))
        ax.text(xy[0], xy[1], ('%i' % i), va='center', ha='center', size=8)
    ax.add_artist(Circle((0, 0), radius=0.5e3, fill=False, color='k',
                         linestyle='--'))

    r_max = 0.5e3
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    fig.savefig('ska1_v7_core.png')
    plt.close(fig)


def plot2():
    fig, ax = _create_figure()

    # ax.add_artist(Circle((0, 0), radius=0.5e3, fill=False, color='0.5',
    #                      linestyle='-'))
    for i, xy in enumerate(zip(x1, y1)):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=False, color='b', lw=1))
        # ax.text(xy[0], xy[1], ('%i' % i), va='center', ha='center', size=8)

    # Rings
    # ax.annotate('Rings', xy=(1.7e3/2**0.5, 1.7e3/2**0.5),
    #             xytext=(3e3, 3e3),
    #             arrowprops=dict(facecolor='k', shrink=0.0001))
    # ax.add_artist(Circle((0, 0), radius=1.7e3, fill=False, color='0.5',
    #                      linestyle='-'))
    for xy in zip(x2, y2):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='r', lw=1))

    # Spiral arms
    ax.add_artist(Circle((0, 0), radius=6.4e3, fill=False, color='0.5',
                         linestyle='--'))
    for xy in zip(x3[::3], y3[::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
    for xy in zip(x3[1::3], y3[1::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
    for xy in zip(x3[2::3], y3[2::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))

    r_max = 6.4e3
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    fig.savefig('ska1_v7_core_area.png')
    plt.close(fig)





def main():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.95)

    # Core
    for i, xy in enumerate(zip(x1, y1)):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='b', lw=1))
        ax.text(xy[0], xy[1], ('%i' % i), va='center', ha='center', size=8)
    ax.add_artist(Circle((0, 0), radius=0.5e3, fill=False, color='0.8',
                         linestyle='--'))

    # Rings
    for xy in zip(x2, y2):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='r', lw=1))
    ax.add_artist(Circle((0, 0), radius=1.7e3, fill=False, color='0.8',
                         linestyle='--'))

    # Spiral arms
    for xy in zip(x3[::3], y3[::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
    for xy in zip(x3[1::3], y3[1::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
    for xy in zip(x3[2::3], y3[2::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
    ax.add_artist(Circle((0, 0), radius=6.4e3, fill=False, color='0.8',
                         linestyle='--'))

    # Outer stations
    for xy in zip(x4, y4):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='k', lw=1))

    r_max = 10e3
    ax.set_xlim(-r_max * 1.1, r_max * 1.1)
    ax.set_ylim(-r_max * 1.1, r_max * 1.1)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    # ax.set_xlim(-2000, 2000)
    # ax.set_ylim(-2000, 2000)
    # plt.savefig('layout.pdf')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    plot1()
    plot2()
    # main()
