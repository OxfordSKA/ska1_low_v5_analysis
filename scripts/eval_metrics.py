# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os.path import join, isdir, isfile
from os import makedirs, listdir
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from math import ceil, floor, log10


def round_up(x):
    return int(10**ceil(log10(x)))

def round_down(x):
    return int(10**floor(log10(x)))

class AnalyseMetrics(object):
    def __init__(self, out_dir):
        if not isdir(out_dir):
            makedirs(out_dir)
        self.out_dir = out_dir
        self.cable_length = dict()

    def analyse(self, tel, tel_r):
        filename = join(self.out_dir, '%s_stations.png' % tel.name)
        if not isfile(filename):
            tel.plot_layout(filename=filename, xy_lim=7e3)

        filename = join(self.out_dir, '%s_cables.png' % tel.name)
        l_ = tel.eval_cable_length(plot=True, plot_filename=filename,
                                   plot_r=7e3)
        self.cable_length[tel_r] = l_

    def save(self, name):
        # All metrics in npz format
        np.savez(join(self.out_dir, '%s_metrics.npz' % name),
                 cable_length=self.cable_length)
        # ASCII CSV table of radius vs cable length
        data = np.array([[k, v] for k, v in self.cable_length.iteritems()])
        data = np.sort(data, axis=0)
        np.savetxt(join(self.out_dir, '%s_cables.txt' % name), data,
                   fmt=b'%.10f %.10f')

    def plot_cable_length_compare(self):
        data = [[k, v] for k, v in self.cable_length.iteritems()]
        data = np.array(data)
        data = np.sort(data, axis=0)
        fig, ax = plt.subplots()
        ax.plot(data[:, 0], data[:, 1], 'x-')
        plt.show()


def compare_cable_lengths(out_dir):
    ref = np.loadtxt(join(out_dir, 'ska1_v5_cables.txt'))
    ref_length = ref[1]

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    files = [f for f in listdir(out_dir)
             if '_cables.txt' in f and
             f.startswith('model')]

    r_min = 1e20
    r_max = 0
    for name in files:
        filename = join(out_dir, name)
        lengths = np.loadtxt(filename)
        r = lengths[:, 0]
        r_min = min(round_down(r.min()), r_min)
        r_max = max(round_up(r.max()), r_max)
        delta_lengths = lengths[:, 1] - ref_length
        ax1.plot(r, delta_lengths / 1e3, '+-', label=name[:7])
    ax1.legend(loc='best')
    ax1.set_xlabel('unwrap radius (m)')
    ax1.set_ylabel('cable length - cable length v5 [=%.2f km] (km)' %
                   (ref_length / 1e3))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    print(r_min, r_max)
    ax1.set_xlim(500, 7000)
    for axis in [ax1.xaxis, ax1.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    ax1.grid()
    # plt.show()
    fig1.savefig(join(out_dir, 'compare_cable_length_increase.png'))
    plt.close(fig1)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    for name in files:
        filename = join(out_dir, name)
        lengths = np.loadtxt(filename)
        r = lengths[:, 0]
        r_min = min(round_down(r.min()), r_min)
        r_max = max(round_up(r.max()), r_max)
        ax1.plot(r, lengths[:, 1] / ref_length, '+-', label=name[:7])
    ax1.legend(loc='best')
    ax1.set_xlabel('unwrap radius (m)')
    ax1.set_ylabel('cable length / cable length v5 [= %.2f km] (km)' %
                   (ref_length / 1e3))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    print(r_min, r_max)
    ax1.set_xlim(500, 7000)
    for axis in [ax1.xaxis, ax1.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    ax1.grid()
    fig1.savefig(join(out_dir, 'compare_cable_length_increase_factor.png'))
    # plt.show()
    plt.close(fig1)

if __name__ == '__main__':
    compare_cable_lengths(join('..', 'results', 'cable_lengths'))
