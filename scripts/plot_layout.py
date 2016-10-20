# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle
from os.path import join


def load_enu():
    # order: core -> rings -> outer -> spirals
    # data = np.loadtxt(join(b'v7_coords', b'ska1_low_v7_enu.txt'))
    data = np.loadtxt(join(b'v7_coords', b'ska1_low_v7.enu.512x4.txt'))
    x = data[:-1, 1]
    y = data[:-1, 2]
    print(x.size)
    assert(x.size == 512)

    num_core = 224
    num_rings = 69
    num_spiral = 93
    num_outer = 512 - num_core - num_rings - num_spiral
    assert(num_outer == 7 * 6 * 3)
    x_outer = x[num_core + num_rings:num_core + num_rings + num_outer]
    y_outer = y[num_core + num_rings:num_core + num_rings + num_outer]
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
    return x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer


def load_wgs84():
    data = np.loadtxt(join(b'v7_coords', b'ska1_low_v7.wgs84.512x4.txt'))
    x = data[:-1, 1]
    y = data[:-1, 2]
    assert (x.size == 512)
    # order: core -> rings -> outer -> spirals
    num_core = 224
    num_rings = 69
    num_spiral = 93
    num_outer = 512 - num_core - num_rings - num_spiral
    assert (num_outer == 7 * 6 * 3)

    x_outer = x[num_core + num_rings:num_core + num_rings + num_outer]
    y_outer = y[num_core + num_rings:num_core + num_rings + num_outer]

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
    return x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer


def write_table():
    x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer = load_wgs84()
    fh = open('table_wgs84.txt', 'w')
    # x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer = load_enu()
    # fh = open('table_enu.txt', 'w')
    fh.write(b'%-10i & %-15s & % -18.8f & % -18.8f \\\ \hline\n'
             % (0, 'Array Centre', 116.7644482, -26.82472208))
    # Core
    id = 1
    for i, xy in enumerate(zip(x1, y1)):
        fh.write(b'%-10i & %-15s & % -18.8f & % -18.8f \\\ \hline\n' %
                 (id, ('C%i' % i), xy[0], xy[1]))
        id += 1
    # Rings
    num_ring = [21, 27, 21]
    color_ = ['r', 'r', 'r']
    for j in range(3):
        i0 = int(np.sum(num_ring[:j]))
        i1 = int(np.sum(num_ring[:j + 1]))
        x2_r = x2[i0:i1]
        y2_r = y2[i0:i1]
        angle_ = np.arctan2(y2_r, x2_r)
        sort_idx = np.argsort(angle_)
        x2_r = x2_r[sort_idx]
        y2_r = y2_r[sort_idx]
        for i, xy in enumerate(zip(x2_r, y2_r)):
            fh.write(b'%-10i & %-15s & % -18.8f & % -18.8f \\\ \hline\n' %
                     (id, ('R%i-%i' % (j, i)), xy[0], xy[1]))
            id += 1
    # Spirals
    for i, xy in enumerate(zip(x3[::3], y3[::3])):
        fh.write(b'%-10i & %-15s & % -18.8f & % -18.8f \\\ \hline\n' %
                 (id, ('S0-%i' % i), xy[0], xy[1]))
        id += 1
    for i, xy in enumerate(zip(x3[1::3], y3[1::3])):
        fh.write(b'%-10i & %-15s & % -18.8f & % -18.8f \\\ \hline\n' %
                 (id, ('E0-%i' % i), xy[0], xy[1]))
        id += 1
    for i, xy in enumerate(zip(x3[2::3], y3[2::3])):
        fh.write(b'%-10i & %-15s & % -18.8f & % -18.8f \\\ \hline\n' %
                 (id, ('N0-%i' % i), xy[0], xy[1]))
        id += 1
    # Outer arms
    for i in range(3):
        i0 = i * 7 * 6
        for j in range(7):
            i1 = i0 + j * 6
            i2 = i1 + 6
            if i == 0:
                label_ = 'E%i' % (j + 10)
            elif i == 1:
                label_ = 'S%i' % (j + 10)
            elif i == 2:
                label_ = 'N%i' % (j + 10)
            for k, xy in enumerate(zip(x_outer[i1:i2], y_outer[i1:i2])):
                fh.write(b'%-10i & %-15s & % -18.8f & % -18.8f \\\ \hline\n' %
                         (id, ('%s-%i' % (label_, k + 1)), xy[0], xy[1]))
                id += 1

    fh.close()


def _create_figure(left=0.1, bottom=0.04):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    fig.subplots_adjust(left=left, right=0.98, bottom=bottom, top=1.0)
    return fig, ax


def plot1():
    x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer = load_enu()
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
    fig.savefig('ska1_v7_core.eps')
    plt.close(fig)


def plot2():
    x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer = load_enu()
    fig, ax = _create_figure(left=0.12)

    # ax.add_artist(Circle((0, 0), radius=0.5e3, fill=False, color='0.5',
    #                      linestyle='-'))
    for i, xy in enumerate(zip(x1, y1)):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=False, color='b', lw=1))
        # ax.text(xy[0], xy[1], ('%i' % i), va='center', ha='center', size=8)

    # Rings
    ax.add_artist(Circle((0, 0), radius=1.7e3, fill=False, color='0.5',
                         linestyle='--'))
    num_ring = [21, 27, 21]
    color_ = ['r', 'r', 'r']
    for j in range(3):
        i0 = int(np.sum(num_ring[:j]))
        i1 = int(np.sum(num_ring[:j + 1]))
        x2_r = x2[i0:i1]
        y2_r = y2[i0:i1]
        angle_ = np.arctan2(y2_r, x2_r)
        sort_idx = np.argsort(angle_)
        x2_r = x2_r[sort_idx]
        y2_r = y2_r[sort_idx]
        for i, xy in enumerate(zip(x2_r, y2_r)):
            ax.add_artist(Circle(xy, radius=35 / 2, fill=True,
                                 color=color_[j], lw=1))
            ax.text(xy[0], xy[1] + 50, ('R%i-%i' % (j, i)),
                    va='center', ha='center', size=8)

    # Spiral arms
    ax.add_artist(Circle((0, 0), radius=6.4e3, fill=False, color='0.5',
                         linestyle='--'))
    for xy in zip(x3[::3], y3[::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
    for xy in zip(x3[1::3], y3[1::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
    for xy in zip(x3[2::3], y3[2::3]):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))

    # Outer stations
    for xy in zip(x4, y4):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='k', lw=1))

    r_max = 2.0e3
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    fig.savefig('ska1_v7_core_rings.eps')
    plt.close(fig)


def plot3():
    x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer = load_enu()
    fig, ax = _create_figure(left=0.12)

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
    for i, xy in enumerate(zip(x3[::3], y3[::3])):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
        # if i > 10 or i == 0 or i == 5:
        #     ax.text(xy[0] - 50, xy[1], ('S0-%i' % i), va='center', ha='right',
        #             size=6)
        if i == 20:
            ax.text(xy[0] + 200, xy[1], 'S0-[0,30]', size=10, va='center')

    for i, xy in enumerate(zip(x3[1::3], y3[1::3])):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
        # if i > 10 or i == 0 or i == 5:
        #     ax.text(xy[0] - 50, xy[1], ('E0-%i' % i), va='center', ha='right',
        #             size=6)
        if i == 20:
            ax.text(xy[0] + 200, xy[1], 'E0-[0,30]', size=10, va='center')

    for i, xy in enumerate(zip(x3[2::3], y3[2::3])):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='g', lw=1))
        # if i > 10 or i == 0 or i == 5:
        #     ax.text(xy[0] - 50, xy[1], ('N0-%i' % i), va='center', ha='right',
        #             size=6)
        if i == 20:
            ax.text(xy[0] + 200, xy[1], 'N0-[0,30]', size=10, va='center')

    # Outer stations
    for xy in zip(x4, y4):
        ax.add_artist(Circle(xy, radius=35 / 2, fill=True, color='k', lw=1))

    r_max = 6.4e3
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    fig.savefig('ska1_v7_core_area.eps')
    plt.close(fig)


def plot4():
    x1, y1, x2, y2, x3, y3, x4, y4, x_outer, y_outer = load_enu()
    fig, ax = _create_figure(left=0.1)
    st_r = (35 / 2) / 1e3
    # ax.add_artist(Circle((0, 0), radius=0.5e3, fill=False, color='0.5',
    #                      linestyle='-'))
    for i, xy in enumerate(zip(x1 / 1e3, y1 / 1e3)):
        ax.add_artist(Circle(xy, radius=st_r, fill=False, color='b', lw=1))
        # ax.text(xy[0], xy[1], ('%i' % i), va='center', ha='center', size=8)

    # Rings
    # ax.annotate('Rings', xy=(1.7e3/2**0.5, 1.7e3/2**0.5),
    #             xytext=(3e3, 3e3),
    #             arrowprops=dict(facecolor='k', shrink=0.0001))
    # ax.add_artist(Circle((0, 0), radius=1.7e3, fill=False, color='0.5',
    #                      linestyle='-'))
    for xy in zip(x2 / 1e3, y2 / 1e3):
        ax.add_artist(Circle(xy, radius=st_r, fill=True, color='r', lw=1))

    # Spiral arms
    ax.add_artist(Circle((0, 0), radius=6.4, fill=False, color='0.5',
                         linestyle='--'))
    for xy in zip(x3[::3] / 1e3, y3[::3] / 1e3):
        ax.add_artist(Circle(xy, radius=st_r, fill=True, color='g', lw=1))
    for xy in zip(x3[1::3] / 1e3, y3[1::3] / 1e3):
        ax.add_artist(Circle(xy, radius=st_r, fill=True, color='g', lw=1))
    for xy in zip(x3[2::3] / 1e3, y3[2::3] / 1e3):
        ax.add_artist(Circle(xy, radius=st_r, fill=True, color='g', lw=1))

    # Outer stations
    # for xy in zip(x4 / 1e3, y4 / 1e3):
    #     ax.add_artist(Circle(xy, radius=st_r, fill=True, color='k', lw=2))
    for i in range(3):
        i0 = i * 7 * 6
        for j in range(7):
            i1 = i0 + j * 6
            i2 = i1 + 6
            for xy in zip(x_outer[i1:i2] / 1e3, y_outer[i1:i2] / 1e3):
                ax.add_artist(Circle(xy, radius=st_r, fill=True, color='k',
                                     lw=3))
            if i == 0:
                label_ = 'E%i' % (j + 10)
            elif i == 1:
                label_ = 'S%i' % (j + 10)
            elif i == 2:
                label_ = 'N%i' % (j + 10)
            ax.text(x_outer[i1] / 1e3, y_outer[i1] / 1e3 + 1,
                    label_, ha='center', va='center', size=8)

    r_max = 40
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.grid(True)
    fig.savefig('ska1_v7_full.eps')
    plt.close(fig)


if __name__ == '__main__':
    plot1()
    plot2()
    plot3()
    plot4()
    write_table()
