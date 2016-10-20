# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os.path import join, isdir, isfile
from os import makedirs, listdir
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from math import ceil, floor, log10
from utilities.telescope_analysis import TelescopeAnalysis
from scipy.io import savemat
from collections import OrderedDict
import re
import pickle
from pprint import pprint


def round_up(x):
    return int(10**ceil(log10(x)))


def round_down(x):
    return int(10**floor(log10(x)))


def _get_option(options, name, default):
    value = default
    if isinstance(options, dict):
        if name in options:
            value = options[name]
    return value


class Metrics(object):
    def __init__(self, out_dir):
        if not isdir(out_dir):
            makedirs(out_dir)
        self.out_dir = out_dir
        self.tel_r = list()
        self.eval_metrics = None
        self.cable_length = dict()
        self.cable_length_2 = dict()
        self.cable_length_3 = dict()
        self.uv_hist = dict()
        self.psf_rms = dict()
        self.psf = dict()

    @staticmethod
    def __write_matlab_clusters(tel, filename):
        # type: (TelescopeAnalysis, str) -> None
        """Write clusters to a MATLAB mat file"""
        centre_x = np.array([])
        centre_y = np.array([])
        points_x = np.array([])
        points_y = np.array([])
        for name in tel.layouts:
            if name == 'ska1_v5':
                continue
            layout = tel.layouts[name]
            centre_x = np.hstack((centre_x, layout['cx']))
            centre_y = np.hstack((centre_y, layout['cy']))
            if points_x.size == 0:
                points_x = layout['x']
                points_y = layout['y']
            else:
                points_x = np.vstack((points_x, layout['x']))
                points_y = np.vstack((points_y, layout['y']))
        savemat(filename, dict(centre_x=centre_x, centre_y=centre_y,
                               antennas_x=points_x, antennas_y=points_y))

    def analyse_telescope(self, tel, tel_r, metrics_list=dict()):
        # type: (TelescopeAnalysis, float) -> None
        """Produce analysis metrics_list for the supplied telescope."""

        if not metrics_list:  # An empty dict evaluates to False.
            return
        self.eval_metrics = metrics_list

        self.tel_r.append(tel_r)

        # -------- ENU layout file --------------------------------------------
        _key = 'layout_enu'
        if _key in metrics_list and metrics_list[_key]:
            tel.save_enu(join(self.out_dir, '%s_enu.txt' % tel.name))

        # -------- MATLAB layout file ----------------------------------------
        _key = 'layout_matlab'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_layout.mat' % tel.name)
            Metrics.__write_matlab_clusters(tel, filename)

        # -------- Layout pickle file -----------------------------------------
        _key = 'layout_pickle'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_layout.p' % tel.name)
            tel.save_pickle(filename)

        # -------- Layout plot ------------------------------------------------
        _key = 'layout_plot'
        if _key in metrics_list and metrics_list[_key]:
            xy_lims = _get_option(metrics_list[_key], 'xy_lims', list())
            # metrics_list[_key].pop('xy_lims', None)
            for xy_lim in xy_lims:
                filename = join(self.out_dir, '%s_stations_%05.1fkm.png' %
                                (tel.name, xy_lim / 1e3))
                tel.plot_layout(filename=filename, xy_lim=xy_lim,
                                **metrics_list[_key])
            if not xy_lims:
                filename = join(self.out_dir, '%s_stations.png' % tel.name)
                tel.plot_layout(filename=filename, **metrics_list[_key])

        # -------- iantconfig files -------------------------------------------
        _key = 'layout_iantconfig'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s' % tel.name)
            tel.save_iantconfig(filename)

        # -------- Simplistic cluster cable length assignment -----------------
        _key = 'cable_length_1'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_cables_1.png' % tel.name)
            l_ = tel.eval_cable_length(plot=True, plot_filename=filename,
                                       plot_r=7.5e3)
            self.cable_length[tel_r] = l_

        # -------- Cluster cable length with attempt at better clustering -----
        _key = 'cable_length_2'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_cables_2.png' % tel.name)
            l_ = tel.eval_cable_length_2(plot=True, plot_filename=filename,
                                         plot_r=7.5e3)
            self.cable_length_2[tel_r] = l_

        # -------- Genetic algorithm for clustering ---------------------------
        _key = 'cable_length_3'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_cables.png' % tel.name)
            l_ = tel.eval_cable_length_3(plot_filename=filename, plot_r=7.5e3)
            self.cable_length_3[tel_r] = l_
            # print('  * cable_length = %.2f km' % (l_/1e3))

        # -------- Generate and plot the uv grid ------------------------------
        _key = 'uv_grid'
        if _key in metrics_list and metrics_list[_key]:
            args = metrics_list[_key]
            tel.grid_cell_size_m = _get_option(args, 'grid_cellsize_m',
                                               tel.station_diameter_m)
            if 'xy_lim' in args:
                for xy_lim in args['xy_lim']:
                    filename = join(self.out_dir, '%s_uv_grid_%03.1fkm.png'
                                    % (tel.name, xy_lim / 1e3))
                    tel.plot_grid(filename, xy_lim=xy_lim)
            else:
                xy_lim = 13e3
                filename = join(self.out_dir, '%s_uv_grid_%03.1fkm.png'
                                % (tel.name, xy_lim / 1e3))
                tel.plot_grid(filename, xy_lim=xy_lim)

        # -------- Generate and plot the uv grid -----------------------------
        _key = 'uv_scatter'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_uv_scatter.png' % tel.name)
            tel.plot_uv_scatter(filename, xy_lim=4e3)

        # -------- Generate and plot the uv histograms (LOG) ------------------
        _key = 'uv_hist_log'
        if _key in metrics_list and metrics_list[_key]:
            args = metrics_list[_key]
            b_max = _get_option(args, 'b_max', 13e3)
            tel.gen_uvw_coords()
            num_bins = int((b_max - tel.station_diameter_m) //
                           tel.station_diameter_m)
            num_bins = min(num_bins, 500)
            num_bins = _get_option(args, 'num_bins', num_bins)
            b_min = _get_option(args, 'b_min', tel.station_diameter_m / 2)

            # Log histograms
            filename = join(self.out_dir, '%s_uv_hist_log_%05.1fkm.png' %
                            (tel.name, b_max / 1e3))
            tel.uv_hist(num_bins=num_bins, filename=filename, log_bins=True,
                        bar=True, b_min=b_min, b_max=b_max)
            filename = join(self.out_dir, '%s_uv_cum_hist_log_%05.1fkm.png' %
                            (tel.name, b_max / 1e3))
            if _get_option(args, 'cum_hist', False):
                tel.uv_cum_hist(filename, log_x=True)
            if not self.uv_hist:
                self.uv_hist[tel.name] = dict()
            self.uv_hist[tel.name]['log_%05.1fkm' % (b_max / 1e3)] = dict(
                hist_n=tel.hist_n, hist_x=tel.hist_x, hist_bins=tel.hist_bins,
                cum_hist_n=tel.cum_hist_n)

        # -------- Generate and plot the uv histograms (LINEAR) ---------------
        _key = 'uv_hist_lin'
        if _key in metrics_list and metrics_list[_key]:
            args = metrics_list[_key]
            b_max = _get_option(args, 'b_max', 13e3)
            tel.gen_uvw_coords()
            num_bins = int((b_max - tel.station_diameter_m) //
                           tel.station_diameter_m)
            num_bins = min(num_bins, 500)
            num_bins = _get_option(args, 'num_bins', num_bins)
            b_min = _get_option(args, 'b_min', tel.station_diameter_m / 2)

            # Lin histograms
            filename = join(self.out_dir, '%s_uv_hist_lin_%05.1fkm.png' %
                            (tel.name, b_max / 1e3))
            make_plot = _get_option(args, 'make_plot', True)
            tel.uv_hist(num_bins=num_bins, filename=filename, log_bins=False,
                        bar=True, b_min=b_min, b_max=b_max, make_plot=make_plot)
            if _get_option(args, 'cum_hist', False):
                filename = join(self.out_dir,
                                '%s_uv_cum_hist_lin_%05.1fkm.png' %
                                (tel.name, b_max / 1e3))
                tel.uv_cum_hist(filename, log_x=True)
            if not self.uv_hist:
                self.uv_hist[tel.name] = dict()
            self.uv_hist[tel.name]['lin_%05.1fkm' % (b_max / 1e3)] = dict(
                hist_n=tel.hist_n, hist_x=tel.hist_x, hist_bins=tel.hist_bins,
                cum_hist_n=tel.cum_hist_n)

        # -------- Generate and plot the uv histograms (LINEAR) ---------------
        _key = 'layout_hist_lin'
        if _key in metrics_list and metrics_list[_key]:
            args = metrics_list[_key]
            b_max = _get_option(args, 'b_max', 13e3)
            tel.gen_uvw_coords()
            num_bins = int((b_max - tel.station_diameter_m) //
                           tel.station_diameter_m)
            num_bins = min(num_bins, 200)
            num_bins = _get_option(args, 'num_bins', num_bins)
            b_min = _get_option(args, 'b_min', tel.station_diameter_m / 2)
            filename = join(self.out_dir, '%s_layout_hist_lin_%05.1fkm.png' %
                            (tel.name, b_max / 1e3))
            tel.layout_hist(num_bins=num_bins, filename=filename,
                            log_bins=False, bar=True, b_min=b_min, b_max=b_max)
            # filename = join(self.out_dir, '%s_layout_cum_hist_lin_%05.1fkm.png' %
            #                 (tel.name, b_max / 1e3))
            # tel.layout_cum_hist(filename, log_x=False)

        # -------- Generate and plot MST network ------------------------------
        _key = 'mst_network'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_network.png' % tel.name)
            tel.network_graph()
            tel.plot_network(filename, plot_r=7e3)
            filename = join(self.out_dir, '%s_network_2.png' % tel.name)
            tel.network_graph_2()
            tel.plot_network_2(filename, plot_r=7e3)

        # -------- Generate and plot PSFRMS -----------------------------------
        _key = 'psf_rms'
        if _key in metrics_list and metrics_list[_key]:
            tel.eval_psf_rms_r(num_bins=20, b_min=500, b_max=10000)
            self.psf_rms[tel.name] = dict()
            self.psf_rms[tel.name]['x'] = tel.psf_rms_r_x
            self.psf_rms[tel.name]['y'] = tel.psf_rms_r

        # -------- Generate and plot UVGAP -----------------------------------
        _key = 'uv_gap'
        if _key in metrics_list and metrics_list[_key]:
            tel.uvgap()

        # -------- Generate and plot PSF -----------------------------------
        _key = 'psf'
        if _key in metrics_list and metrics_list[_key]:
            filename = join(self.out_dir, '%s_psf' % tel.name)
            tel.eval_psf(filename_root=filename, plot1d=True, plot2d=True,
                         fov_deg=5.0, im_size=4096, num_bins=256)
            self.psf[tel.name] = dict()
            # self.psf[tel.name]['image'] = tel.psf
            self.psf[tel.name]['fov'] = tel.psf_fov_deg
            self.psf[tel.name]['1d_r'] = tel.psf_1d['r']
            self.psf[tel.name]['1d_min'] = tel.psf_1d['min']
            self.psf[tel.name]['1d_max'] = tel.psf_1d['max']
            self.psf[tel.name]['1d_std'] = tel.psf_1d['std']
            self.psf[tel.name]['1d_rms'] = tel.psf_1d['rms']
            self.psf[tel.name]['1d_mean'] = tel.psf_1d['mean']
            self.psf[tel.name]['1d_abs_mean'] = tel.psf_1d['abs_mean']
            self.psf[tel.name]['1d_abs_max'] = tel.psf_1d['abs_max']

    def plot_comparisons(self, psf_1d=False):
        """Generate comparison plots."""
        if self.psf_rms:  # Checks if self.psf_rms is empty
            fig, ax = plt.subplots(figsize=(8, 8))
            for tel in self.psf_rms:
                ax.plot(self.psf_rms[tel]['x'], self.psf_rms[tel]['y'])
            plt.show()

        if psf_1d and self.psf:
            fig, ax = plt.subplots(figsize=(8, 8))
            for tel in self.psf:
                x = self.psf[tel]['1d_r']
                y = self.psf[tel]['1d_abs_max']
                ax.plot(x, y, label=tel)
            ax.set_yscale('log')
            ax.legend(loc='best')
            plt.show()

    def save_results(self, name):
        """Save metrics to disk"""
        # metrics in npz format
        # filename = join(self.out_dir, '%s_metrics.npz' % name)
        # np.savez(filename, tel_r=self.tel_r, cable_length=self.cable_length,
        #          cable_length_2=self.cable_length_2, uv_hist=self.uv_hist)

        if self.cable_length:  # Empty dict() evaluate to False
            # ASCII CSV table of radius vs cable length
            filename = join(self.out_dir, '%s_cables.txt' % name)
            data = np.array([[k, v] for k, v in
                             self.cable_length.iteritems()])
            data = np.sort(data, axis=0)
            np.savetxt(filename, data, fmt=b'%.10f %.10f')

        if self.cable_length_2:  # Empty dict() evaluate to False
            # ASCII CSV table of radius vs cable length
            filename = join(self.out_dir, '%s_cables_2.txt' % name)
            data = np.array([[k, v] for k, v in
                             self.cable_length_2.iteritems()])
            data = np.sort(data, axis=0)
            np.savetxt(filename, data, fmt=b'%.10f %.10f')

        if self.cable_length_3:  # Empty dict() evaluates to False
            filename = join(self.out_dir, '%s_cables.txt' % name)
            data = np.array([[k, v] for k, v in
                             self.cable_length_3.iteritems()])
            data = np.sort(data, axis=0)
            np.savetxt(filename, data, fmt=b'%.10f %.10f')

        # Save a pickle with the PSF comparison info.
        if self.psf:
            filename = join(self.out_dir, '%s_psf.p' % name)
            pickle.dump(self.psf, open(filename, 'wb'))

        # Save a pickle of uv hist data.
        if self.uv_hist:
            filename = join(self.out_dir, '%s_uv_hist.p' % name)
            pickle.dump(self.uv_hist, open(filename, 'wb'))

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

    @staticmethod
    def compare_psf_1d(results_dir):
        plot_style = [
            {'color': 'k', 'marker': 'none', 'lw': 1.5},
            {'color': 'b', 'marker': 'none', 'lw': 1.5},
            {'color': 'g', 'marker': 'none', 'lw': 1.5},
            {'color': 'r', 'marker': 'none', 'lw': 1.5},
            {'color': 'c', 'marker': 'none', 'lw': 1.5},
            {'color': 'm', 'marker': 'none', 'lw': 1.5}
            # {'color': 'c', 'marker': 'none', 'lw': 1.5},
            # {'color': 'm', 'marker': 'none', 'lw': 1.5},
            # {'color': 'y', 'marker': 'none', 'lw': 1.5}
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.1,  bottom=0.1, right=0.95, top=0.95)
        idx = 0
        # models = ['ska1_v5', 'model01', 'model02', 'model03']
        # models = ['ska1_v5',
        #           'model01', 'model02', 'model03', 'model04', 'model05']
        models = ['ska1_v5', 'model07', 'model03', 'model08']
        unwrap_levels = ['r08']
        for model in models:
            # Load saved psf results.
            filename = join(results_dir, '%s_psf.p' % model)
            psf = pickle.load(open(filename, 'rb'))

            # Plot comparison of PSF profiles.
            for tel in psf:
                if tel == 'ska1_v5' or any(s in tel for s in unwrap_levels):
                    print(tel)
                    if tel == 'ska1_v5':
                        label = 'SKA1 v5'
                    if 'model' in tel:
                        label = 'Model ' + str(re.search(r'\d+', tel).group())
                    style = plot_style[idx % len(plot_style)]
                    x = psf[tel]['1d_r']
                    # y = psf[tel]['1d_max']
                    # ax.plot(x, y, color=style['color'],
                    #         linestyle='-',
                    #         label=label + ' max')
                    # y = psf[tel]['1d_min']
                    # ax.plot(x, y, color=style['color'],
                    #         linestyle='--',
                    #         label=label + ' min')
                    y = psf[tel]['1d_abs_mean']
                    ax.plot(x, y, color=style['color'],
                            linestyle='-',
                            label=label + ' mean')
                    # y = psf[tel]['1d_rms']
                    # ax.plot(x, y, color=style['color'],
                    #         linestyle=':',
                    #         label=label + ' rms')
                    idx += 1

        ax.legend(loc='best', fontsize='x-small')
        ax.set_xlabel('lm distance')
        ax.set_ylabel('PSF mean')
        ax.set_xlim(0, x.max())
        ax.set_ylim(4e-4, 4e-2)
        # ax.set_ylim(1e-3, 0.02)
        # ax.set_ylim(-0.02, 0.02)
        ax.set_yscale('log')
        ax.grid()
        fig.savefig(join(results_dir, 'compare_mean_%s.png' %
                         '_'.join(unwrap_levels)))
        # plt.show()
        plt.close(fig)

    @staticmethod
    def compare_cum_hist(results_dir, log_axis=False):
        plot_style = [
            {'color': 'k', 'marker': 'none', 'lw': 1.5},
            {'color': 'c', 'marker': 'none', 'lw': 1.5},
            {'color': 'r', 'marker': 'none', 'lw': 1.5},
            {'color': 'y', 'marker': 'none', 'lw': 1.5}
        ]
        if log_axis:
            axis_type = 'log'
        else:
            axis_type = 'lin'

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.1,  bottom=0.1, right=0.95, top=0.95)
        idx = 0
        models = ['ska1_v5', 'model01', 'model02', 'model03']
        unwrap_levels = ['r08']
        # models = ['ska1_v5', 'model01']
        for model in models:
            # Load saved psf results.
            filename = join(results_dir, '%s_uv_hist.p' % model)
            psf = pickle.load(open(filename, 'rb'))

            # Plot comparison of PSF profiles.
            for tel in psf:
                if tel == 'ska1_v5' or any(s in tel for s in unwrap_levels):
                    print(tel)
                    style = plot_style[idx % len(plot_style)]
                    x = psf[tel][axis_type]['hist_x']
                    y = psf[tel][axis_type]['cum_hist_n']
                    ax.plot(x, y, color=style['color'],
                            linestyle='-',
                            label=tel + ' max')
                    idx += 1

        ax.legend(loc='best', fontsize='small', ncol=1)
        ax.set_xlabel('Baseline length (m)')
        ax.set_ylabel('Fraction of baselines')
        # ax.set_xlim(0, x.max())
        # ax.set_ylim(0, 1.05)
        if log_axis:
            ax.set_yscale('log')
            ax.set_xscale('log')
        ax.grid()
        fig.savefig(join(results_dir, 'compare_cum_hist_%s_%s.png' %
                         (axis_type, '_'.join(unwrap_levels))))
        plt.show()
        plt.close(fig)

    @staticmethod
    def compare_hist(results_dir, log_axis=False):
        """Compare histograms for several models found in the same results
        folder.
        """
        plot_style = [
            {'color': 'k', 'marker': 'none', 'lw': 1.5},
            {'color': 'r', 'marker': 'none', 'lw': 1.5}
        ]
        if log_axis:
            axis_type = 'log'
        else:
            axis_type = 'lin'

        unwrap_levels = ['r08']

        for i in [3, 7, 8]:
            models = ['ska1_v5', 'model%02i' % i]
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
            idx = 0
            for model in models:
                # Load saved psf results.
                filename = join(results_dir, '%s_uv_hist.p' % model)
                print('-- Loading: %s' % filename)
                psf = pickle.load(open(filename, 'rb'))

                # Plot comparison of PSF profiles.
                for tel in psf:
                    if tel == 'ska1_v5' or any(s in tel for s in unwrap_levels):
                        style = plot_style[idx % len(plot_style)]
                        x = psf[tel][axis_type]['hist_x']
                        y = psf[tel][axis_type]['hist_n']
                        cut_idx = np.argmax(x > 5e3)
                        x = x[:cut_idx]
                        y = y[:cut_idx]
                        ax.plot(x, y, color=style['color'],
                                linestyle='-',
                                label=tel + ' max')
                        idx += 1

            ax.legend(loc='best', fontsize='small', ncol=1)
            ax.set_xlabel('Baseline length (m)')
            ax.set_ylabel('Number of baselines per bin')
            ax.set_xlim(10, 14e3)
            # ax.set_ylim(500, ax.get_ylim()[1])
            if log_axis:
                ax.set_yscale('log')
                ax.set_xscale('log')
            ax.grid()
            fig.savefig(join(results_dir, 'compare_hist_%s_%s_%s.png' %
                             (axis_type, '_'.join(unwrap_levels), models[1])))
            # plt.show()
            plt.close(fig)

    @staticmethod
    def compare_hist_2(results_dir_root):
        """Compare pairs of histograms for snapshots and 4 hour observations"""
        plot_style = [
            {'color': 'k', 'marker': 'none', 'lw': 1.5},
            {'color': 'r', 'marker': 'none', 'lw': 1.5}
        ]
        unwrap_levels = ['r08']

        fig, axes = plt.subplots(figsize=(8, 10), nrows=5, ncols=2, sharex=True,
                                 sharey=False)
        fig.subplots_adjust(left=0.1, bottom=0.08, right=0.98, top=0.98,
                            hspace=0.05, wspace=0.15)

        print(axes.shape)

        results_dirs = [results_dir_root + '_0h', results_dir_root + '_4h']

        fig.text(0.5, 0.02, r'Baseline length (m)', ha='center')
        fig.text(0.01, 0.5, r'Baseline count per bin', va='center',
                 rotation='vertical')

        # Loop over models
        for i in range(7, 8):
            models = ['ska1_v5', 'model%02i' % (i + 1)]
            # Loop over 0h or 4h
            for j in range(1):
                idx = 0
                ax = axes[i, j]
                for model in models:
                    # Load saved psf results.
                    filename = join(results_dirs[j], '%s_uv_hist.p' % model)
                    print('-- Loading: %s' % filename)
                    hist = pickle.load(open(filename, 'rb'))
                    # Plot comparison of PSF profiles.
                    for tel in hist:
                        if tel == 'ska1_v5' or any(s in tel for s in unwrap_levels):
                            style = plot_style[idx % len(plot_style)]
                            x = hist[tel]['log']['hist_x']
                            y = hist[tel]['log']['hist_n']
                            cut_idx = np.argmax(x > 5e3)
                            x = x[:cut_idx]
                            y = y[:cut_idx]
                            ax.plot(x, y, color=style['color'],
                                    linestyle='-',
                                    label=tel + ' max')
                            idx += 1
                length_str = 'snapshot' if j == 0 else '4 hours'
                ax.text(0.02, 0.95, ('SKA1 v5 vs. Model %i\n' % (i + 1)) + length_str,
                        weight='bold',
                        ha='left', va='top', transform=ax.transAxes,
                        fontsize='small')
                # ax.legend(loc='best', fontsize='small', ncol=1)
                # ax.set_xlabel('Baseline length (m)')
                # ax.set_ylabel('Number of baselines per bin')
                ax.set_xlim(10, 1e4)
                if j == 1:
                    ax.set_ylim(1.5e3)
                if j == 0:
                    ax.set_ylim(5)
                ax.set_yscale('log')
                ax.set_xscale('log')
                # if i == 4:
                #     ax.set_xlabel('Baseline length (m)')
                # ax.set_ylabel('Baseline count per bin')
                ax.grid()
        plt.show(True)
        fig.savefig(join('compare_hist_%s.png' %
                         ('_'.join(unwrap_levels))))
        plt.close(fig)
