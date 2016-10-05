# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import re
from numpy.matlib import repmat
import os
from os.path import join
from . import telescope
from . import sensitivity
from pyuvwsim import evaluate_baseline_uvw_ha_dec, convert_enu_to_ecef
from math import pi, radians, degrees, ceil, asin, sin, log10, floor, sqrt
from oskar.imager import Imager
from astropy import constants as const
from astropy.visualization import (HistEqStretch, SqrtStretch,
                                   LogStretch)
from astropy.visualization.mpl_normalize import ImageNormalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from collections import Counter


class TelescopeAnalysis(telescope.Telescope):
    """Telescope analysis functions."""
    def __init__(self, name=''):
        telescope.Telescope.__init__(self, name)
        self.dec_deg = 0
        self.grid_cell_size_m = self.station_diameter_m
        self.freq_hz = 100e6
        self.bandwidth_hz = 100e3
        self.obs_length_h = 0
        self.num_times = 1
        self.uu_m = None
        self.vv_m = None
        self.ww_m = None
        self.r_uv_m = None
        self.uv_grid = None
        self.tree = None
        self.tree_length = 0
        self.tree_2 = None
        self.tree_length_2 = 0
        self.hist_bins = None
        self.hist_n = None
        self.hist_x = None
        self.cum_hist_n = None
        self.psf_rms = 0
        self.psf_rms_r = None
        self.psf_rms_r_x = None
        self.psf = None
        self.psf_fov_deg = 0
        self.psf_1d = dict()

    def clear_layouts(self):
        super(TelescopeAnalysis, self).clear_layouts()
        self.uu_m = None
        self.vv_m = None
        self.ww_m = None
        self.r_uv_m = None
        self.uv_grid = None
        self.tree = None
        self.tree_length = 0
        self.tree_2 = None
        self.tree_length_2 = 0
        self.hist_n = None
        self.hist_x = None
        self.psf_rms = 0
        self.psf_rms_r = None
        self.psf_rms_r_x = None
        self.psf = None
        self.psf_fov_deg = 0
        self.psf_1d = dict()

    def gen_uvw_coords(self):
        """Generate uvw coordinates"""
        x, y, z = self.get_coords_enu()
        x, y, z = convert_enu_to_ecef(x, y, z, radians(self.lon_deg),
                                      radians(self.lat_deg), self.alt_m)
        num_stations = x.shape[0]
        num_baselines = num_stations * (num_stations - 1) // 2
        n = num_baselines * self.num_times
        self.uu_m, self.vv_m, self.ww_m = np.zeros(n), np.zeros(n), np.zeros(n)
        ha_off = ((self.obs_length_h / 2) / 24) * (2 * pi)
        for i, ha in enumerate(np.linspace(-ha_off, ha_off, self.num_times)):
            uu_, vv_, ww_ = evaluate_baseline_uvw_ha_dec(
                x, y, z, ha - radians(self.lon_deg), radians(self.dec_deg))
            self.uu_m[i * num_baselines: (i + 1) * num_baselines] = uu_
            self.vv_m[i * num_baselines: (i + 1) * num_baselines] = vv_
            self.ww_m[i * num_baselines: (i + 1) * num_baselines] = ww_
        self.r_uv_m = (self.uu_m**2 + self.vv_m**2)**0.5

    def num_coords(self):
        return self.uu_m.size if self.uu_m is not None else 0

    def grid_uvw_coords(self):
        if self.uu_m is None:
            self.gen_uvw_coords()
        b_max = self.r_uv_m.max()

        grid_size = int(ceil(b_max / self.grid_cell_size_m)) * 2 + \
                    self.station_diameter_m
        if grid_size % 2 == 1:
            grid_size += 1
        wavelength = const.c.value / self.freq_hz
        # delta theta = 1 / (n * delta u)
        cell_lm = 1.0 / (grid_size * (self.grid_cell_size_m / wavelength))
        lm_max = grid_size * sin(cell_lm) / 2
        uv_grid_fov_deg = degrees(asin(lm_max)) * 2
        imager = Imager('single')
        imager.set_grid_kernel('pillbox', 1,  1)
        imager.set_size(grid_size)
        imager.set_fov(uv_grid_fov_deg)
        weight_ = np.ones_like(self.uu_m)
        amps_ = np.ones_like(self.uu_m, dtype='c8')
        self.uv_grid = np.zeros((grid_size, grid_size), dtype='c8')
        norm = imager.update_plane(self.uu_m / wavelength,
                                   self.vv_m / wavelength,
                                   self.ww_m / wavelength,
                                   amps_, weight_, self.uv_grid,
                                   plane_norm=0.0)
        norm = imager.update_plane(-self.uu_m / wavelength,
                                   -self.vv_m / wavelength,
                                   -self.ww_m / wavelength, amps_,
                                   weight_, self.uv_grid, plane_norm=norm)
        if not int(norm) == self.uu_m.shape[0] * 2:
            raise RuntimeError('Gridding uv coordinates failed, '
                               'grid sum = %i != number of points gridded = %i'
                               % (norm, self.uu_m.shape[0] * 2))

    def plot_grid(self, filename=None, show=False, plot_radii=[],
                  xy_lim=None):
        if self.uv_grid is None:
            self.grid_uvw_coords()
        grid_size = self.uv_grid.shape[0]
        wavelength = const.c.value / self.freq_hz
        fov_rad = Imager.uv_cellsize_to_fov(self.grid_cell_size_m / wavelength,
                                            grid_size)
        extent = Imager.grid_extent_wavelengths(degrees(fov_rad), grid_size)
        extent = np.array(extent) * wavelength

        fig, ax = plt.subplots(figsize=(8, 8), ncols=1, nrows=1)
        fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                            wspace=0.2, hspace=0.2)
        image = self.uv_grid.real
        options = dict(interpolation='nearest', cmap='gray_r', extent=extent,
                       origin='lower')
        im = ax.imshow(image, norm=ImageNormalize(stretch=LogStretch()),
                       **options)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.03)
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.set_label('baselines per pixel')
        cbar.ax.tick_params(labelsize='small')
        ticks = np.arange(5) * round(image.max() / 5)
        ticks = np.append(ticks, image.max())
        cbar.set_ticks(ticks, update_ticks=True)

        for r in plot_radii:
            ax.add_artist(plt.Circle((0, 0), r, fill=False, color='r'))
        ax.set_xlabel('uu (m)')
        ax.set_ylabel('vv (m)')
        ax.grid(True)
        if xy_lim is not None:
            ax.set_xlim(-xy_lim, xy_lim)
            ax.set_ylim(-xy_lim, xy_lim)
        if filename is not None:
            label = ''
            if 'ska1_v5' in filename:
                label = 'SKA1 v5'
            if 'model' in filename:
                label = 'Model ' + str(
                    re.search(r'\d+',
                              os.path.basename(filename)).group())
            ax.text(0.02, 0.95, label, weight='bold',
                    transform=ax.transAxes)
            fig.savefig(filename)
        if show:
            plt.show()
        if filename is not None or show:
            plt.close(fig)
        else:
            return fig

    def uv_hist(self, num_bins=100, b_min=None, b_max=None, make_plot=True,
                log_bins=True, bar=False, filename=None):
        if self.uu_m is None:
            self.gen_uvw_coords()

        b_max = self.r_uv_m.max() if b_max is None else b_max
        b_min = self.r_uv_m.min() if b_min is None else b_min

        if log_bins:
            bins = np.logspace(log10(b_min), log10(b_max), num_bins + 1)
        else:
            bins = np.linspace(b_min, b_max, num_bins + 1)

        self.hist_n, _ = np.histogram(self.r_uv_m, bins=bins, density=False)
        self.hist_x = (bins[1:] + bins[:-1]) / 2
        self.hist_bins = bins

        if make_plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            if bar:
                ax.bar(self.hist_x, self.hist_n, width=np.diff(bins),
                       alpha=0.8, align='center', lw=0, color='0.2')
            else:
                ax.plot(self.hist_x, self.hist_n)
            if log_bins:
                ax.set_xscale('log')
            ax.set_xlabel('baseline length (m)')
            ax.set_ylabel('Number of baselines per bin')
            ax.set_xlim(0, b_max * 1.1)
            ax.grid()
            if filename is not None:
                fig.savefig(filename)
            else:
                plt.show()
            plt.close(fig)

    def uv_cum_hist(self, filename, log_x=False):
        if self.hist_n is None:
            self.uv_hist(make_plot=False)

        self.cum_hist_n = np.cumsum(self.hist_n) / self.uu_m.size
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.hist_bins[1:], self.cum_hist_n, 'k-')
        i_half = np.argmax(self.cum_hist_n >= 0.5)
        bin_width_i_half = self.hist_bins[i_half+1] - self.hist_bins[i_half]
        xy = (self.hist_bins[i_half], 0.5)
        xy_text = (self.hist_bins[i_half] * 2, 0.45)
        ax.annotate('%.1f $\pm$ %.1f m' %
                    (self.hist_bins[i_half], bin_width_i_half),
                    xy=xy, xytext=xy_text, ha='left', va='top',
                    arrowprops=dict(facecolor='k', width=0.0001, headwidth=5),
                    fontsize='small')
        ax.set_xlabel('baseline length (m)')
        ax.set_ylabel('Fraction of baselines')
        ax.grid()
        ax.set_ylim(0, 1.05)
        if log_x:
            ax.set_xlim(self.hist_bins[0], self.hist_bins[-1])
            ax.set_xscale('log')
        else:
            ax.set_xlim(0, self.hist_bins[-1])
        ax.plot(ax.get_xlim(), [0.5, 0.5], '--', color='0.5')
        fig.savefig(filename)
        plt.close(fig)

    def uv_sensitivity(self, num_bins=100, b_min=None, b_max=None,
                       log_bins=True):
        if self.hist_n is None:
            self.uv_hist(num_bins, b_min, b_max, log_bins=log_bins,
                         make_plot=False)
        t_int = self.obs_length_h * 3600 if self.obs_length_h > 0 else 1
        b_max = self.r_uv_m.max() if b_max is None else b_max
        beam_solid_angle, _ = sensitivity.beam_solid_angle(self.freq_hz, b_max)
        sigma_t = sensitivity.brightness_temperature_sensitivity(
            self.freq_hz, beam_solid_angle, t_int,
            self.bandwidth_hz)
        print(sigma_t)
        # TODO(BM) check that the sum of hist n == number of baselines
        #  --- ie that the telescope is normalised correctly.
        bar = False
        fig, ax = plt.subplots(figsize=(8, 8))
        y = sigma_t / np.sqrt(self.hist_n)
        if bar:
            ax.bar(self.hist_x, y, width=np.diff(self.hist_bins),
                   alpha=0.8, align='center', lw=0.5)
        else:
            ax.plot(self.hist_x, y)
        if log_bins:
            ax.set_xscale('log')
        ax.set_xlabel('baseline length (m)')
        ax.set_ylabel('Brightness sensitivity (K)')
        ax.set_yscale('log')
        ax.set_xlim(0, b_max * 1.1)
        plt.show()
        plt.close(fig)

    def network_graph(self):
        x, y, _ = self.get_coords_enu(include_core=False)
        coords = np.transpose(np.vstack([x, y]))
        self.tree = minimum_spanning_tree(squareform(pdist(coords))).toarray()
        self.tree_length = np.sum(self.tree)

    def network_graph_2(self):
        x, y = self.get_centres_enu()
        coords = np.transpose(np.vstack([x, y]))
        self.tree_2 = minimum_spanning_tree(squareform(pdist(coords))).toarray()
        self.tree_length_2 = np.sum(self.tree_2)

    def plot_network(self, filename, plot_r=None):
        if self.tree is None:
            self.network_graph()
        x, y, _ = self.get_coords_enu(include_core=False)
        cx, cy = self.get_centres_enu()
        # self.plot_layout(mpl_ax=ax)
        fig, ax = plt.subplots()
        for xy in zip(x, y):
            ax.add_artist(plt.Circle(xy, self.station_diameter_m/2,
                                     fill=False))
        ax.plot(cx, cy, 'r+')
        for i in range(y.size):
            for j in range(x.size):
                if self.tree[i, j] > 0:
                    ax.plot([x[i], x[j]], [y[i], y[j]], 'g-', alpha=0.5,
                            lw=1.0)
        ax.grid()
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.text(0.05, 0.95, 'Total: %.2f km' % (self.tree_length / 1e3),
                transform=ax.transAxes, va='top', ha='left',
                fontsize='small')
        if plot_r:
            ax.set_xlim(-plot_r, plot_r)
            ax.set_ylim(-plot_r, plot_r)
        ax.set_aspect('equal')
        fig.savefig(filename)
        plt.close(fig)

    def plot_network_2(self, filename, plot_r=None):
        if self.tree is None:
            self.network_graph()
        x, y, _ = self.get_coords_enu(include_core=False)
        cx, cy = self.get_centres_enu()
        # self.plot_layout(mpl_ax=ax)
        fig, ax = plt.subplots()
        for xy in zip(x, y):
            ax.add_artist(plt.Circle(xy, self.station_diameter_m/2,
                                     fill=False))
        ax.plot(cx, cy, 'r+')
        for i in range(cy.size):
            for j in range(cx.size):
                if self.tree_2[i, j] > 0:
                    ax.plot([cx[i], cx[j]], [cy[i], cy[j]], 'g-', alpha=0.5,
                            lw=1.0)
        ax.grid()
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.text(0.05, 0.95, 'Total: %.2f km' % (self.tree_length_2 / 1e3),
                transform=ax.transAxes, va='top', ha='left',
                fontsize='small')
        if plot_r:
            ax.set_xlim(-plot_r, plot_r)
            ax.set_ylim(-plot_r, plot_r)
        ax.set_aspect('equal')
        fig.savefig(filename)
        plt.close(fig)

    def eval_psf_rms(self, num_bins=100, b_min=None, b_max=None):
        """Single PSFRMS for the grid"""
        if self.uv_grid is None:
            self.grid_uvw_coords()
        self.psf_rms = (np.sqrt(np.sum(self.uv_grid.real**2)) /
                        (self.uu_m.size * 2))
        return self.psf_rms

    def eval_psf_rms_r(self, num_bins=100, b_min=None, b_max=None):
        """PSFRMS radial profile"""
        if self.uv_grid is None:
            self.grid_uvw_coords()
        grid_size = self.uv_grid.shape[0]
        gx, gy = Imager.grid_pixels(self.grid_cell_size_m, grid_size)
        gr = (gx**2 + gy**2)**0.5

        b_max = self.r_uv_m.max() if b_max is None else b_max
        b_min = self.r_uv_m.min() if b_min is None else b_min
        r_bins = np.logspace(log10(b_min), log10(b_max), num_bins + 1)
        self.psf_rms_r = np.zeros(num_bins)
        for i in range(num_bins):
            pixels = self.uv_grid[np.where(gr <= r_bins[i + 1])]
            uv_idx = np.where(self.r_uv_m <= r_bins[i + 1])[0]
            uv_count = uv_idx.shape[0] * 2
            self.psf_rms_r[i] = 1.0 if uv_count == 0 else \
                np.sqrt(np.sum(pixels.real**2)) / uv_count
        self.psf_rms_r_x = r_bins[1:]

        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.plot(self.psf_rms_r_x, self.psf_rms_r)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_xlabel('radius (m)')
        # ax.set_ylabel('PSFRMS')
        # ax.set_xlim(10**floor(log10(self.psf_rms_r_x[0])),
        #             10**ceil(log10(self.psf_rms_r_x[-1])))
        # ax.set_ylim(10**floor(log10(self.psf_rms_r.min())), 1.05)
        # ax.grid(True)
        # plt.show()

    def uvgap(self):
        pass

    def eval_psf(self, im_size=None, fov_deg=None, plot2d=False, plot1d=True,
                 filename_root=None, num_bins=100):
        """Evaluate and plot the PSF."""
        if self.uu_m is None:
            self.gen_uvw_coords()

        # Work out a usable grid size.
        if im_size is None:
            b_max = self.r_uv_m.max()
            grid_size = int(ceil(b_max / self.grid_cell_size_m)) * 2 + \
                self.station_diameter_m
            if grid_size % 2 == 1:
                grid_size += 1
        else:
            grid_size = im_size

        # Work out the FoV
        wavelength = const.c.value / self.freq_hz
        if fov_deg is None:
            cellsize_wavelengths = self.grid_cell_size_m / wavelength
            fov_rad = Imager.uv_cellsize_to_fov(cellsize_wavelengths,
                                                grid_size)
            fov_deg = degrees(fov_rad)
        else:
            fov_deg = fov_deg

        uu = self.uu_m / wavelength
        vv = self.vv_m / wavelength
        ww = self.ww_m / wavelength
        amp = np.ones_like(uu, dtype='c8')
        psf = Imager.make_image(uu, vv, ww, amp, fov_deg, grid_size)
        extent = Imager.image_extent_lm(fov_deg, grid_size)
        self.psf = psf
        self.psf_fov_deg = fov_deg

        # --- Plotting ----
        if plot2d:
            fig, ax = plt.subplots(figsize=(8, 8))
            norm = SymLogNorm(linthresh=0.01, linscale=1.0, vmin=-0.02,
                              vmax=1.0, clip=False)
            # opts = dict(interpolation='nearest', origin='lower', cmap='gray_r',
            #             extent=extent, norm=norm)
            opts = dict(interpolation='nearest', origin='lower', cmap='gray_r',
                        extent=extent, norm=norm)
            im = ax.imshow(psf, **opts)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.03)
            cbar = ax.figure.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize='small')
            ax.set_xlabel('l')
            ax.set_ylabel('m')
            if filename_root:
                label = ''
                if 'ska1_v5' in filename_root:
                    label = 'SKA1 v5'
                if 'model' in filename_root:
                    label = 'Model ' + str(
                        re.search(r'\d+',
                                  os.path.basename(filename_root)).group())
                ax.text(0.02, 0.95, label, color='white', weight='bold',
                        transform=ax.transAxes)
                plt.savefig('%s_2d.png' % filename_root)
            else:
                plt.show()
            plt.close(fig)

        if plot1d:
            l, m = Imager.image_pixels(self.psf_fov_deg, grid_size)
            r_lm = (l**2 + m**2)**0.5
            r_lm = r_lm.flatten()
            idx_sorted = np.argsort(r_lm)
            r_lm = r_lm[idx_sorted]
            psf_1d = self.psf.flatten()[idx_sorted]
            psf_hwhm = (wavelength / (r_lm[-1] * 2.0)) / 2

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(r_lm, psf_1d, 'k.', ms=2, alpha=0.1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            fig.savefig('TEST_psf1d.png')
            plt.close(fig)

            psf_1d_mean = np.zeros(num_bins)
            psf_1d_abs_mean = np.zeros(num_bins)
            psf_1d_abs_max = np.zeros(num_bins)
            psf_1d_min = np.zeros(num_bins)
            psf_1d_max = np.zeros(num_bins)
            psf_1d_std = np.zeros(num_bins)
            psf_1d_rms = np.zeros(num_bins)
            bin_edges = np.linspace(r_lm[0], r_lm[-1] * 2**(-0.5), num_bins + 1)
            # bin_edges = np.logspace(log10(r_lm[1]), log10(r_lm[-1]),
            #                         num_bins + 1)
            psf_1d_bin_r = (bin_edges[1:] + bin_edges[:-1]) / 2
            bin_idx = np.digitize(r_lm, bin_edges)
            for i in range(1, num_bins + 1):
                values = psf_1d[bin_idx == i]
                if values.size > 0:
                    psf_1d_mean[i - 1] = np.mean(values)
                    psf_1d_abs_mean[i - 1] = np.mean(np.abs(values))
                    psf_1d_abs_max[i - 1] = np.max(np.abs(values))
                    psf_1d_min[i - 1] = np.min(values)
                    psf_1d_max[i - 1] = np.max(values)
                    psf_1d_std[i - 1] = np.std(values)
                    psf_1d_rms[i - 1] = np.mean(values**2)**0.5

            self.psf_1d['r'] = psf_1d_bin_r
            self.psf_1d['min'] = psf_1d_min
            self.psf_1d['max'] = psf_1d_max
            self.psf_1d['mean'] = psf_1d_mean
            self.psf_1d['std'] = psf_1d_std
            self.psf_1d['rms'] = psf_1d_rms
            self.psf_1d['abs_mean'] = psf_1d_abs_mean
            self.psf_1d['abs_max'] = psf_1d_abs_max

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(psf_1d_bin_r, psf_1d_abs_mean, '-', c='b', lw=1, label='abs mean')
            ax.plot(psf_1d_bin_r, psf_1d_abs_max, '-', c='r', lw=1, label='abs max')
            ax.plot(psf_1d_bin_r, psf_1d_std, '-', c='g', lw=1, label='std')
            # ax.set_ylim(-0.1, 0.5)
            # ax.set_xlim(0, psf_1d_bin_r[-1] / 2**0.5)
            # ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1)
            ax.set_xlim(0, psf_1d_bin_r[-1] / 2**0.5)
            ax.set_xlabel('PSF radius (direction cosines)')
            # ax.set_ylabel('PSF amplitude')
            ax.set_title('Azimuthally averaged PSF (FoV: %.2f)' % fov_deg)
            ax.legend()
            if filename_root:
                plt.savefig('%s_1d.png' % filename_root)
            else:
                plt.show()
            plt.close(fig)

    def eval_cable_length(self, plot=False, plot_filename=None, plot_r=None):
        """Get cable lengths using simple direct links to cluster centres."""
        # FIXME(BM) check bug where there are not 6 coordinates in a cluster ?!
        num_clusters = 0
        cluster_cable_length = 0.0
        expended_cable_length = 0.0
        r_max_expanded = 0.0
        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')
        for key in self.layouts.keys():
            if key == 'ska1_v5':
                layout = self.layouts[key]
                if plot:
                    for p in zip(layout['x'], layout['y']):
                        ax.add_artist(plt.Circle(p, self.station_diameter_m / 2,
                                                 fill=False, color='k',
                                                 alpha=0.2))
            elif '_cluster' in key:
                num_clusters += 1
                layout = self.layouts[key]
                cx, cy = (layout['cx'], layout['cy'])
                x, y = (layout['x'], layout['y'])
                print(x.size)
                if plot:
                    for p in zip(x, y):
                        ax.add_artist(plt.Circle(p, self.station_diameter_m / 2,
                                                 fill=False, color='k'))
                    ax.plot(cx, cy, 'r+', ms=10)
                cluster_total = 0.0
                for x_, y_ in zip(x, y):
                    dx = x_ - cx
                    dy = y_ - cy
                    r = (dx**2 + dy**2)**0.5
                    cluster_total += r
                    if plot:
                        ax.plot([x_, cx], [y_, cy], ':', color='0.5')
                if plot:
                    ax.text(cx + self.station_diameter_m * 6,
                            cy + self.station_diameter_m * 6,
                            '%.1f' % cluster_total,
                            va='center', ha='center',
                            fontsize='xx-small')
                cluster_cable_length += cluster_total
                # print('Custer %i = %.2f m' % (num_clusters, cluster_total))
            else:
                layout = self.layouts[key]
                cx, cy = (layout['cx'], layout['cy'])
                x, y = (layout['x'], layout['y'])
                cr = (cx**2 + cy**2)**0.5
                r_max_expanded = max(cr, r_max_expanded)
                if plot:
                    for p in zip(x, y):
                        ax.add_artist(plt.Circle(p, self.station_diameter_m / 2,
                                                 fill=False, color='k'))
                    ax.plot(cx, cy, 'r+', ms=10)
                cluster_total = 0.0
                for x_, y_ in zip(x, y):
                    dx = x_ - cx
                    dy = y_ - cy
                    r = (dx**2 + dy**2)**0.5
                    cluster_total += r
                    if plot:
                        ax.plot([x_, cx], [y_, cy], ':', color='0.5')
                if plot:
                    ax.text(cx + self.station_diameter_m * 6,
                            cy + self.station_diameter_m * 6,
                            '%.1f' % cluster_total,
                            va='center', ha='center',
                            fontsize='xx-small')
                expended_cable_length += cluster_total
        total_cable_length = cluster_cable_length + expended_cable_length
        if plot:
            # cx, cy = self.get_centres_enu()
            # cr = (cx**2 + cy**2)**0.5
            # ax.add_artist(plt.Circle((0, 0), cr.max(), fill=False, lw=0.5,
            #                          color='0.3', linestyle=':'))
            # ax.add_artist(plt.Circle((0, 0), r_max_expanded, fill=False, lw=0.5,
            #                          color='g', linestyle='-', alpha=0.5))
            ax.set_xlabel('east (m)')
            ax.set_ylabel('north (m)')
            ax.text(0.05, 0.95, 'Total: %.2f km' % (total_cable_length / 1e3),
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize='small')
            if plot_r is not None:
                ax.set_xlim(-plot_r, plot_r)
                ax.set_ylim(-plot_r, plot_r)
            if plot_filename is not None:
                fig.savefig(plot_filename)
            else:
                plt.show()
            plt.close(fig)
        # print(cluster_cable_length, num_clusters)
        return total_cable_length

    def eval_cable_length_2(self, cluster_size=6, plot=False,
                            plot_filename=None, plot_r=None):
        # Get cluster centres
        cx, cy = self.get_centres_enu()

        # Get all station coordinates not in the core
        x, y = np.array(list()), np.array(list())
        cluster_count = 0
        for name in self.layouts:
            layout = self.layouts[name]
            if name == 'ska1_v5':  # This key is the core
                continue
            # print(name, layout['x'].shape)
            x = np.hstack([x, layout['x']])
            y = np.hstack([y, layout['y']])
            cluster_count += 1
        x = np.array(x)
        y = np.array(y)

        # Find 6 stations for each cluster.
        # Algorithm:
        #   Loop over cluster centres 6 times.
        #   On each loop each cluster grabs its closet point.

        # Radial distance of each point from each cluster centre.
        dcr = np.zeros((cx.size, x.size))
        xy_idx = np.arange(x.size, dtype=np.int)
        for c in range(cx.size):
            dcr[c, :] = ((cx[c] - x)**2 + (cy[c] - y)**2)**0.5

        total_cable_length = 0
        c_idx = np.empty((cx.size, cluster_size), dtype=np.int)
        # Loop over stations (s) in cluster
        for s in range(cluster_size):
            # Loop over clusters (c)
            for c in range(cx.size):
                # Add the closest point to the cluster (by radius)
                ii = dcr[c, :].argmin()
                total_cable_length += dcr[c, ii]
                c_idx[c, s] = xy_idx[ii]
                # Remove this point from contention
                dcr = np.delete(dcr, ii, axis=1)
                xy_idx = np.delete(xy_idx, ii, axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        for p in zip(x, y):
            ax.add_artist(plt.Circle(p, self.station_diameter_m/2,
                                     fill=False, color='k'))
        ax.plot(cx, cy, 'r+', ms=10)
        for c in range(cx.size):
            cluster_cable_length = 0
            for s in range(cluster_size):
                dx = cx[c] - x[c_idx[c, s]]
                dy = cy[c] - y[c_idx[c, s]]
                dr = (dx**2 + dy**2)**0.5
                cluster_cable_length += dr
                ax.plot([cx[c], x[c_idx[c, s]]],
                        [cy[c], y[c_idx[c, s]]], ':', color='0.5')
            ax.text(cx[c] + self.station_diameter_m * 6,
                    cy[c] + self.station_diameter_m * 6,
                    '%.1f' % cluster_cable_length,
                    va='center', ha='center',
                    fontsize='xx-small')

        for p in zip(self.layouts['ska1_v5']['x'], self.layouts['ska1_v5']['y']):
            ax.add_artist(plt.Circle(p, self.station_diameter_m / 2,
                                     fill=False, color='k',
                                     alpha=0.2))

        ax.set_xlabel('east (m)')
        ax.set_ylabel('north (m)')
        ax.text(0.05, 0.95, 'Total: %.2f km' % (total_cable_length / 1e3),
                transform=ax.transAxes, va='top', ha='left',
                fontsize='small')
        if plot_r is not None:
            ax.set_xlim(-plot_r, plot_r)
            ax.set_ylim(-plot_r, plot_r)
        if plot_filename is not None:
            fig.savefig(plot_filename)
        else:
            plt.show()
        plt.close(fig)
        return total_cable_length

    @staticmethod
    def __get_cable_lengths(cx, cy, x, y):
        total_distance = 0
        cluster_distance = np.zeros((x.size, cx.size), order='Fortran')

        num_per_cluster = x.size // cx.size

        # Loop over clusters and work out distance to each antenna
        for i in range(cx.size):
            dx = x - cx[i]
            dy = y - cy[i]
            cluster_distance[:, i] = (dx**2 + dy**2)**0.5

        # Loop over clusters to evaluate the cluster index of each antenna
        # and the total cable length for the telescope.
        i2 = 0
        for i in range(cx.size):
            i1 = i2
            i2 += num_per_cluster
            total_distance += np.sum(cluster_distance[i1:i2, i])

        return total_distance, cluster_distance

    def __plot_clustering(self, cx, cy, x, y, plot_r, filename=None):
        """Plot clustering between cluster centres cx, cy and positions x, y"""
        cable_length = self.__get_cable_lengths(cx, cy, x, y)[0] / 1e3
        fig, ax = plt.subplots()
        for xy in zip(x, y):
            ax.add_artist(plt.Circle(xy, self.station_diameter_m/2,
                                     fill=False))
        cluster_size = x.size // cx.size
        for j in range(cx.size):
            overlength_count = 0
            for i in range(cluster_size):
                x_ = cx[j] - x[j * cluster_size + i]
                y_ = cy[j] - y[j * cluster_size + i]
                link_length = (x_**2 + y_**2)**0.5
                if link_length > 1e3:
                    overlength_count += 1
                    ax.plot([cx[j], x[j * cluster_size + i]],
                            [cy[j], y[j * cluster_size + i]], '-', color='c')
                # else:
                #     ax.plot([cx[j], x[j * cluster_size + i]],
                #             [cy[j], y[j * cluster_size + i]], ':', color='0.5')
            ax.text(cx[j], cy[j], ('%i' % overlength_count))
        ax.plot(cx, cy, 'r+')
        ax.set_aspect('equal')
        ax.text(0.05, 0.95, 'Total: %.2f km' % cable_length,
                transform=ax.transAxes, va='top', ha='left',
                fontsize='small')
        ax.set_xlim(-plot_r, plot_r)
        ax.set_ylim(-plot_r, plot_r)
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        if filename is not None:
            fig.savefig(filename)
        else:
            plt.show()
        plt.show()
        plt.close(fig)

    def eval_cable_length_3(self, plot_filename=None, plot_r=None):
        """Cable length using Stef's genetic algorithm."""
        cx, cy = self.get_centres_enu()
        x, y, _ = self.get_coords_enu(include_core=False)
        # self.__plot_clustering(cx, cy, x, y, plot_r=7e3)
        cable_length0, cluster_dist = self.__get_cable_lengths(cx, cy, x, y)

        num_in_generation = 1000
        num_mutations = 1
        num_survivors = 3
        num_generations = 200
        num_stations = x.size
        num_clusters = cx.size
        num_per_cluster = x.size // cx.size

        index_list = np.zeros((x.size, num_survivors), dtype=np.int,
                              order='Fortran')
        cable_lengths = np.zeros(num_survivors)

        # Initialise first generation of survivors (== initial parents).
        # FIXME(BM) i don't think the intention is for all of the first gen to
        #           be identical - probably just a bug in the matlab ...
        cluster_index = repmat(range(num_clusters), num_per_cluster,
                               1).T.flatten()
        for i in range(num_survivors):
            index_list[:, i] = np.arange(0, num_stations, dtype=np.int)
            cable_lengths[i], _ = self.__get_cable_lengths(cx, cy, x, y)

        child_index_list = np.zeros((num_stations, num_in_generation),
                                    dtype=np.int, order='Fortran')

        # Number of children per survivor
        nq = (num_in_generation - 1) // num_survivors
        # Remainder
        nr = (num_in_generation - 1) % num_survivors

        for jgen in range(num_generations):
            i2 = 0
            for j in range(num_survivors):
                i1 = i2
                i2 = (j + 1) * nq + min((j + 1), nr)

                for i in range(i1, i2):
                    child_index_list[:, i] = index_list[:, j]

                    for k in range(num_mutations):
                        ia = np.random.randint(num_stations)
                        ib = np.random.randint(num_stations)

                        if ia != ib:
                            child_index_list[ia, i] = index_list[ib, j]
                            child_index_list[ib, i] = index_list[ia, j]

                            iia = child_index_list[ia, i]
                            iib = child_index_list[ib, i]

                            jza = cluster_index[ia]
                            jzb = cluster_index[ib]

                            add1 = cluster_dist[iib, jzb] + cluster_dist[iia, jza]
                            add2 = -cluster_dist[iib, jza] - cluster_dist[iia, jzb]

                            fseqi = cable_lengths[j] + add1 + add2

                            if fseqi < cable_lengths[num_survivors-1]:
                                cable_lengths[num_survivors-1] = fseqi
                                index_list[:, num_survivors-1] = child_index_list[:, i]
                                sorted_idx = np.argsort(cable_lengths)
                                cable_lengths = cable_lengths[sorted_idx]
                                index_list = index_list[:, sorted_idx]

        final_index_list = index_list[:, 0]
        self.__plot_clustering(cx, cy, x[final_index_list], y[final_index_list],
                               7.5e3, plot_filename)
        return cable_lengths[0]


class SKA1_low_analysis(TelescopeAnalysis):
    def __init__(self, name=''):
        TelescopeAnalysis.__init__(self, name)
        self.lon_deg = 116.63128900
        self.lat_deg = -26.69702400

