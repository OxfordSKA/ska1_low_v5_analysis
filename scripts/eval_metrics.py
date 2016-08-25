# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os.path import join, isdir, isfile
from os import makedirs, listdir
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from math import ceil, floor, log10
from utilities.analysis import TelescopeAnalysis
from scipy.io import savemat
from collections import OrderedDict


def round_up(x):
    return int(10**ceil(log10(x)))


def round_down(x):
    return int(10**floor(log10(x)))


class Metrics(object):
    def __init__(self, out_dir):
        if not isdir(out_dir):
            makedirs(out_dir)
        self.out_dir = out_dir
        self.tel_r = list()
        self.cable_length = dict()
        self.cable_length_2 = dict()
        self.uv_hist = dict()

    @staticmethod
    def __write_matlab_clusters(tel, filename):
        """Write clusters to a MATLAB mat file"""
        # type: (TelescopeAnalysis, str) -> None
        coords = np.array([])
        centre_x = np.array([])
        centre_y = np.array([])
        points_x = np.array([])
        points_y = np.array([])
        for name in tel.layouts:
            if name == 'ska1_v5':
                continue
            layout = tel.layouts[name]
            # -----------
            # coords = np.hstack((coords, layout['cx']))
            # coords = np.hstack((coords, layout['cy']))
            # coords = np.hstack((coords, layout['x']))
            # coords = np.hstack((coords, layout['y']))
            # -----------
            # coords_ = np.array([])
            # coords_ = np.hstack((coords_, layout['cx']))
            # coords_ = np.hstack((coords_, layout['cy']))
            # coords_ = np.hstack((coords_, layout['x']))
            # coords_ = np.hstack((coords_, layout['y']))
            # if coords.size == 0:
            #     coords = coords_
            # else:
            #     coords = np.vstack((coords, coords_))
            # -----------
            centre_x = np.hstack((centre_x, layout['cx']))
            centre_y = np.hstack((centre_y, layout['cy']))
            if points_x.size == 0:
                points_x = layout['x']
                points_y = layout['y']
            else:
                points_x = np.vstack((points_x, layout['x']))
                points_y = np.vstack((points_y, layout['y']))

        # savemat(filename, dict(coords=coords))
        savemat(filename, dict(centre_x=centre_x, centre_y=centre_y,
                               antennas_x=points_x, antennas_y=points_y))

    def analyse_telescope(self, tel, tel_r):
        # type: (TelescopeAnalysis, float) -> None

        self.tel_r.append(tel_r)
        tel.save_enu(join(self.out_dir, '%s_enu.txt' % tel.name))

        # filename = '%s_clusters.mat' % tel.name
        # Metrics.__write_matlab_clusters(tel, filename)

        # filename = join(self.out_dir, '%s_stations.png' % tel.name)
        # if not isfile(filename):
        #     tel.plot_layout(filename=filename, xy_lim=7e3,
        #                     show_decorations=False)

        # # Simplistic cluster cable length assignment
        # filename = join(self.out_dir, '%s_cables.png' % tel.name)
        # l_ = tel.eval_cable_length(plot=True, plot_filename=filename,
        #                            plot_r=7e3)
        # self.cable_length[tel_r] = l_

        # # Cluster cable length with attempt at better clustering
        # filename = join(self.out_dir, '%s_cables_2.png' % tel.name)
        # l_ = tel.eval_cable_length_2(plot=True, plot_filename=filename,
        #                              plot_r=7e3)
        # self.cable_length_2[tel_r] = l_
        #
        # filename = join(self.out_dir, '%s_uv_grid.png' % tel.name)
        # tel.plot_grid(filename, xy_lim=13e3)

        # tel.gen_uvw_coords()
        # # num_bins = int((13e3 - tel.station_diameter_m) //
        # #                tel.station_diameter_m)
        # num_bins = 100
        # filename = join(self.out_dir, '%s_uv_hist_log.png' % tel.name)
        # tel.uv_hist(num_bins=num_bins, filename=filename, log_bins=True,
        #             bar=True, b_min=tel.station_diameter_m / 2,
        #             b_max=13e3)
        # filename = join(self.out_dir, '%s_uv_cum_hist_log.png' % tel.name)
        # tel.uv_cum_hist(filename, log_x=True)
        # self.uv_hist[tel.name] = dict(log=dict())
        # self.uv_hist[tel.name]['log'] = dict(hist_n=tel.hist_n,
        #                                      hist_x=tel.hist_x,
        #                                      hist_bins=tel.hist_bins,
        #                                      cum_hist_n=tel.cum_hist_n)
        #
        # filename = join(self.out_dir, '%s_uv_hist_lin.png' % tel.name)
        # tel.uv_hist(num_bins=num_bins, filename=filename, log_bins=False,
        #             bar=True, b_min=tel.station_diameter_m / 2,
        #             b_max=13e3)
        # filename = join(self.out_dir, '%s_uv_cum_hist_lin.png' % tel.name)
        # tel.uv_cum_hist(filename, log_x=False)
        # self.uv_hist[tel.name]['lin'] = dict(hist_n=tel.hist_n,
        #                                      hist_x=tel.hist_x,
        #                                      hist_bins=tel.hist_bins,
        #                                      cum_hist_n=tel.cum_hist_n)

        # filename = join(self.out_dir, '%s_network.png' % tel.name)
        # if not isfile(filename):
        #     tel.network_graph()
        #     tel.plot_network(filename)

        # TODO(BM)
        # Comparison of histogram and cumulative histogram (overplotted lines (not bars))
        # psf (2d, radial profile, PSFRMS)
        # uvgap?

    def save_results(self, name):
        # All metrics in npz format
        filename = join(self.out_dir, '%s_metrics.npz' % name)
        np.savez(filename, tel_r=self.tel_r, cable_length=self.cable_length,
                 cable_length_2=self.cable_length_2, uv_hist=self.uv_hist)

        if self.cable_length:  # Empty dict() evaluate to False
            # ASCII CSV table of radius vs cable length
            filename = join(self.out_dir, '%s_cables.txt' % name)
            if not isfile(filename):
                data = np.array([[k, v] for k, v in self.cable_length.iteritems()])
                data = np.sort(data, axis=0)
                np.savetxt(filename, data, fmt=b'%.10f %.10f')

        if self.cable_length_2:  # Empty dict() evaluate to False
            # ASCII CSV table of radius vs cable length
            filename = join(self.out_dir, '%s_cables_2.txt' % name)
            if not isfile(filename):
                data = np.array([[k, v] for k, v in self.cable_length_2.iteritems()])
                data = np.sort(data, axis=0)
                np.savetxt(filename, data, fmt=b'%.10f %.10f')

    def plot_cable_length_compare(self):
        data = [[k, v] for k, v in self.cable_length.iteritems()]
        data = np.array(data)
        data = np.sort(data, axis=0)
        fig, ax = plt.subplots()
        ax.plot(data[:, 0], data[:, 1], 'x-')
        plt.show()

    @staticmethod
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

    @staticmethod
    def compare_uv_hist(data_file, filename, log_x=True, log_y=False):
        """Compare uv histograms"""
        metrics = np.load(data_file)
        uv_hist = metrics['uv_hist'][()]
        uv_hist = OrderedDict(sorted(uv_hist.items()))
        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, sharex=True)
        for i, model in enumerate(uv_hist):
            if i not in (0, 4, 8):
                continue
            if log_x:
                hist = uv_hist[model]['log']
            else:
                hist = uv_hist[model]['lin']
            ax.plot(hist['hist_x'], hist['hist_n'], '-', label=model)
        ax.legend(loc='best')
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('Baseline length (m)')
        ax.set_ylabel('Baseline count per bin')
        ax.grid()
        fig.savefig(filename)
        plt.close(fig)

    @staticmethod
    def compare_uv_cum_hist(data_file, filename, log_x=True):
        """Compare uv histograms"""
        metrics = np.load(data_file)
        uv_hist = metrics['uv_hist'][()]
        uv_hist = OrderedDict(sorted(uv_hist.items()))
        print(uv_hist.keys())
        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, sharex=True)
        for i, model in enumerate(uv_hist):
            if i not in (0, 4, 8):
                continue
            if log_x:
                hist = uv_hist[model]['log']
            else:
                hist = uv_hist[model]['lin']
            print(hist.keys())
            ax.plot(hist['hist_bins'][1:], hist['cum_hist_n'], '-', label=model)
        ax.legend(loc='best')
        if log_x:
            ax.set_xscale('log')
        ax.plot(ax.get_xlim(), [0.5, 0.5], '--', color='0.5')
        ax.set_xlabel('Baseline length (m)')
        ax.set_ylabel('Fraction of baselines')
        ax.grid()
        fig.savefig(filename)
        plt.close(fig)


if __name__ == '__main__':
    Metrics.compare_cable_lengths(join('..', 'results', 'cable_lengths'))
