"""Script for unwrapping of v5 clusters."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import os
import shutil
import numpy as np
from numpy.random import randint
from utilities.telescope import Telescope, taylor_win
from utilities.telescope_analysis import SKA1_low_analysis
from utilities.eval_metrics import Metrics
from os.path import join, isdir
from os import makedirs
from pprint import pprint


def analyse_v5(out_dir='TEMP_results', obs_length_h=0, num_times=1,
               eval_metrics=dict()):
    """Generate and analyse reference v5 layout."""
    # -------------- Options --------------------------------------------------
    name = 'v5_baseline'  # Name of the telescope model (prefix in file names)
    # outer_radius = 6400 + 90
    # -------------------------------------------------------------------------
    if not isdir(out_dir):
        makedirs(out_dir)

    # Current SKA1 V5 design.
    tel = SKA1_low_analysis(name)
    # tel.add_ska1_v5(r_max=outer_radius)
    tel.add_ska1_v5()
    tel.obs_length_h = obs_length_h
    tel.num_times = num_times
    tel.dec_deg = tel.lat_deg

    metrics = Metrics(out_dir)
    metrics.analyse_telescope(tel, 0.0, eval_metrics)
    # metrics.save_results(name)
    print(tel.name, tel.num_stations())
    return tel, metrics


class AnalyseUnwrapV5(object):
    """Class to unwrap v5 clusters and analyse the result."""
    def __init__(self, out_dir='TEMP_results', remove_existing_results=True,
                 obs_length_h=0, num_times=1, eval_metrics=dict()):
        self.out_dir = out_dir

        if remove_existing_results and os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        self.obs_length_h = obs_length_h
        self.num_times = num_times
        self.eval_metrics = eval_metrics

        # Generate new telescopes based on v5 by expanding each station cluster
        self.core_radius = 500
        self.cluster_radius = 90
        self.outer_radius = 6400
        self.stations_per_cluster = 6

        # Parameters for the log spiral cluster replacement.
        self.b = 0.515
        self.theta0_deg = -48
        self.start_inner = 417.8
        self.num_arms = 3
        self.d_theta = 360 / self.num_arms

        # Get v5 cluster positions
        self.cluster_x, self.cluster_y, self.arm_index = \
            Telescope.cluster_centres_ska_v5(0, self.outer_radius)

        # Get cluster radii (every 3rd index as 3 arms have common radii)
        self.cluster_r = (self.cluster_x[::3]**2 + self.cluster_y[::3]**2)**0.5

        # Get theta separation between cluster rings for the inner and outer
        # log spirals of ska v5 (clusters 0 to 1 and 4 to 5 along the first arm)
        self.delta_theta_deg_inner = Telescope.delta_theta(
            self.cluster_x[0], self.cluster_y[0],
            self.cluster_x[3], self.cluster_y[3],
            self.start_inner, self.b)
        self.delta_theta_deg_outer = Telescope.delta_theta(
            self.cluster_x[12], self.cluster_y[12],
            self.cluster_x[15], self.cluster_y[15],
            self.start_inner, self.b)

        # Convert to theta offset about the cluster centre for the log spiral
        # unpacking.
        self.delta_theta_deg_inner *= (5 / 12)
        self.delta_theta_deg_outer *= (5 / 12)

    def _add_v5_core_clusters(self, name, i, add_core=True):
        """Adds v5 core and clusters not being replaced"""
        # Create the telescope model and add the core (from v5)
        tel = SKA1_low_analysis('%s_r%02i' % (name, i))
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        print('-- Telescope: %s' % tel.name)
        if add_core:
            tel.add_ska1_v5(r_max=self.core_radius)

        # Add the SKA1 v5 clusters we are not replacing
        if i < self.cluster_r.size - 1:
            tel.add_ska1_v5(self.cluster_r[i + 1] - self.cluster_radius,
                            self.outer_radius + self.cluster_radius)
        return tel

    @staticmethod
    def _taper_r_profile(r, amps, taper_r_min=0):
        """Nearest neighbour matching. FIXME(BM) double check this"""
        r = np.asarray(r)
        idx = np.round(
            ((r - taper_r_min) / (1 - taper_r_min)) * (amps.size - 1))
        values = np.asarray(amps[idx.astype(np.int)])
        values[r < taper_r_min] = 1.0
        return values

    def model03(self, name='model03', add_core=True, r_values=None):
        """Model03 == replace clusters with radial arcs and log spirals"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)
        print (self.cluster_r)
        print (self.core_radius)

        # Loop over cluster radii
        for i in range(self.cluster_r.size):
            if r_values is not None and i not in r_values:
                continue
            # Create the telescope model (v5 core & clusters not being replaced)
            tel = self._add_v5_core_clusters(name, i, add_core)
            tel.add_ska1_v5(r_min=6450)

            # Loop over cluster radii we are replacing
            for j in range(i + 1):
                delta_theta_deg = self.delta_theta_deg_inner if j <= 3 else \
                    self.delta_theta_deg_outer
                for k in range(self.num_arms):
                    idx = self.num_arms * j + k
                    if j >= 5:
                        tel.add_log_spiral_section(
                            self.stations_per_cluster, self.start_inner,
                            self.cluster_x[idx], self.cluster_y[idx],
                            self.b, delta_theta_deg,
                            self.theta0_deg + self.arm_index[idx] *
                            self.d_theta)
                    else:
                        tel.add_circular_arc(self.stations_per_cluster,
                                             self.cluster_x[idx],
                                             self.cluster_y[idx], self.d_theta)
            metrics.analyse_telescope(tel, self.cluster_r[i],
                                      self.eval_metrics)

        metrics.save_results(name)
        metrics.plot_comparisons()
        return tel, metrics

    def model06(self, name='model06', add_core=True, r_values=None):
        """***** This function is just for testing *****"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)

        # TODO(BM) hybrid of model1 with a custom generated core
        # and avoiding cable lengths > 1km in arm clusters.
        # For the log spiral arms also generate from a single arm and rotate
        # with stride 3 to generate arms.  (put 162 - 18? in arms) and
        # (224 + 18 in the core) to recover some short baselines.
        from math import log, exp
        import matplotlib.pyplot as plt

        def taper_func_1(r):
            return 1 - (r / 2)**1.5

        def taper_func_2(r, hwhm=1.5, taper_r_min=0):
            r = np.asarray(r)
            c = hwhm / (2 * log(2))**0.5
            values = np.exp(-((r - taper_r_min) / (1 - taper_r_min))**2 /
                            2 * c**2)
            values[r < taper_r_min] = 1.0
            return values

        def taper_r_profile(r, amps, taper_r_min=0):
            """Nearest neighbour matching. FIXME(BM) double check this"""
            r = np.asarray(r)
            idx = np.round(((r - taper_r_min) / (1 - taper_r_min)) * (amps.size - 1))
            values = np.asarray(amps[idx.astype(np.int)])
            values[r < taper_r_min] = 1.0
            return values

        def get_taper_profile(taper_func, r, **kwargs):
            if kwargs is not None:
                t = taper_func(r, **kwargs)
            else:
                t = taper_func(r)
            return t

        tel = SKA1_low_analysis(name)
        sll = -28
        n_taylor = 10000
        r = np.linspace(0, 1, 100)
        t1 = get_taper_profile(taper_func_1, r)
        t2 = get_taper_profile(taper_func_2, r, hwhm=1.0)
        t2a = get_taper_profile(taper_func_2, r, hwhm=1.0, taper_r_min=0.25)
        t3 = get_taper_profile(taper_r_profile, r,
                               amps=taylor_win(n_taylor, sll))
        t4 = get_taper_profile(taper_r_profile, r,
                               amps=taylor_win(n_taylor, sll), taper_r_min=0.25)

        fig, ax = plt.subplots()
        ax.plot(r, tel.station_diameter_m / t1, 'k-', label='Power law')
        ax.plot(r, tel.station_diameter_m / t2, 'b-', label='Gaussian')
        ax.plot(r, tel.station_diameter_m / t2a, 'b--', label='Gaussian rmin')
        ax.plot(r, tel.station_diameter_m / t3, 'r-', label='Taylor %i' % sll)
        ax.plot(r, tel.station_diameter_m / t4, 'r--', label='Taylor rmin %i'
                                                             % sll)
        ax.set_xlabel('Fractional radius')
        ax.set_ylabel('Minimum station separation (m)')
        ax.grid()
        ax.set_ylim(34)
        ax.legend(loc='best')
        fig.savefig('station_min_dist.png')
        # plt.show()
        plt.close(fig)

        core_radius_m = 480

        tel.num_trials = 10
        tel.trial_timeout_s = 30.0
        tel.seed = 24183655
        args = dict(amps=taylor_win(n_taylor, sll), taper_r_min=0.25)
        tel.add_tapered_core(224 + 18, core_radius_m,
                             taper_r_profile, **args)
        tel.plot_layout(filename=name + '.png',
                        show_decorations=False, plot_radii=[500])
        print('final seed =', tel.layouts['tapered_core']['info']['final_seed'])
        print(tel.seed)

        tel.add_log_spiral_2(20, 480, 6000, 0.515, 3, 'inner_arms', 0)
        x1, y1, _ = tel.get_coords_enu()

        tel_v5 = SKA1_low_analysis('ska1_v5')
        tel_v5.add_ska1_v5(r_max=6400)
        x2, y2, _ = tel_v5.get_coords_enu()

        # Side by side comparison to ska v5 core
        fig, axes = plt.subplots(figsize=(16, 8), ncols=2)
        for xy in zip(x1, y1):
            axes[0].add_artist(plt.Circle(xy, tel.station_diameter_m/2,
                                          fill=True, color='k', alpha=0.5))
        for xy in zip(x2, y2):
            axes[1].add_artist(plt.Circle(xy, tel.station_diameter_m/2,
                                          fill=True, color='k', alpha=0.5))
        for ax in axes:
            ax.add_artist(plt.Circle((0, 0), 500, fill=False, color='r'))
            ax.set_aspect('equal')
            ax.set_xlim(-600, 600)
            ax.set_ylim(-600, 600)
        plt.show()
        plt.close(fig)

        # Side by side comparison to ska v5 core
        fig, axes = plt.subplots(figsize=(16, 8), ncols=2)
        for xy in zip(x1, y1):
            axes[0].add_artist(plt.Circle(xy, tel.station_diameter_m/2,
                                          fill=True, color='k', alpha=0.5))
        for xy in zip(x2, y2):
            axes[1].add_artist(plt.Circle(xy, tel.station_diameter_m/2,
                                          fill=True, color='k', alpha=0.5))
        for ax in axes:
            ax.add_artist(plt.Circle((0, 0), 500, fill=False, color='r'))
            ax.set_aspect('equal')
            ax.set_xlim(-7e3, 7e3)
            ax.set_ylim(-7e3, 7e3)
        plt.show()
        plt.close(fig)

        # TODO(BM) add cluster centres into the spiral arms...

        tel.plot_layout(show_decorations=True, xy_lim=2.5e3,
                        plot_radii=[500, 6400])
        tel.plot_layout(show_decorations=True, xy_lim=7e3,
                        plot_radii=[500, 6400])

    def model07(self, name='model07', add_core=True):
        """Proposed SKA v6 configuration option 1

        4 rings of 21 stations (84 total)
            r0 = 500m, r5 = 1700m with no ring at r0
        72 station arms from r0 = 1700m with no point at 1700 to r21 at 6.4 km
         (ie 6 stations symmetric about 6.4km)
           Note(BM) might have to shrink this a bit to match 1km LV reticulation
        core of 224 + 6 stations (uniform or tapered up to 500m)
        """
        metrics = Metrics(self.out_dir)

        tel = SKA1_low_analysis(name + '_r08')
        tel.station_diameter_m = 40
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg

        # Core
        core_radius_m = 480

        # Inner rings
        num_rings = 5
        num_per_ring = 17
        ring_r0 = 580
        ring_r5 = 1700
        ring_radii = np.logspace(np.log10(ring_r0),
                                 np.log10(ring_r5),
                                 num_rings)
        print('ring radii   =', ring_radii)
        print('ring spacing = ', np.diff(ring_radii))

        # Arms
        arm_r0 = ring_r5 + core_radius_m / 2
        arm_r1 = 6400

        # ============== Core
        tel.add_ska1_v5(r_max=500)
        tel.layouts['ska1_v5']['x'] *= 40 / 35
        tel.layouts['ska1_v5']['y'] *= 40 / 35

        # ============== Rings
        np.random.seed(1)
        for i, r in enumerate(ring_radii[0:2]):
            tel.add_ring(num_per_ring, r, delta_theta=randint(0, 360))

        # metrics.analyse_telescope(tel, 0, self.eval_metrics)
        # return tel

        # ============= Spiral arms
        # tel.add_log_spiral_2(25, arm_r0, arm_r1, 0.515, 3, 'inner_arms', 0)
        tel.add_log_spiral_3(25, self.start_inner, arm_r0, arm_r1, 0.515,
                             self.num_arms, self.theta0_deg)

        # # ============= Add outer arms
        # coords = np.loadtxt(join('utilities', 'data', 'v5_enu.txt'))
        # x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        # r = (x**2 + y**2)**0.5
        # idx = np.where(r >= 6.5e3)
        # x, y, z = x[idx], y[idx], z[idx]
        # tel.layouts['outer_arms'] = dict(x=x, y=y, z=z)
        # # tel.add_ska1_v5(6400, 9e50)
        # # =============

        # tel.add_symmetric_log_spiral(25, arm_r0, arm_r1, 0.515, 3,
        #                              'inner_arms', self.delta_theta_deg_inner)

        x, _, _ = tel.get_coords_enu()
        print('total stations =', x.size)

        # Plotting layout
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        # tel.plot_layout()

        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()
        return tel

    def model08(self, name='model08', add_core=True):
        """Proposed SKA v6 configuration option 1

        4 rings of 21 stations (84 total)
            r0 = 500m, r5 = 1700m with no ring at r0
        72 station arms from r0 = 1700m with no point at 1700 to r21 at 6.4 km
         (ie 6 stations symmmetric about 6.4km)
           Note(BM) might have to shrink this a bit to match 1km LV reticulation
        core of 224 + 6 stations (uniform or tapered up to 500m)
        """
        metrics = Metrics(self.out_dir)

        # TODO(BM) hybrid of model1 with a custom generated core
        # and avoiding cable lengths > 1km in arm clusters.
        # For the log spiral arms also generate from a single arm and rotate
        # with stride 3 to generate arms.  (put 162 - 18? in arms) and
        # (224 + 18 in the core) to recover some short baselines.
        from math import log, exp
        import matplotlib.pyplot as plt

        tel = SKA1_low_analysis(name + '_r08')
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg

        # Core
        core_radius_m = 480
        sll = -24
        n_taylor = 10000
        tel.num_trials = 2
        tel.trial_timeout_s = 30.0

        # Inner rings
        num_rings = 5
        num_per_ring = 17
        ring_r0 = 580
        ring_r5 = 1700
        ring_radii = np.logspace(np.log10(ring_r0),
                                 np.log10(ring_r5),
                                 num_rings)
        print('ring radii   =', ring_radii)
        print('ring spacing = ', np.diff(ring_radii))

        # Arms
        # arm_r0 = 1805
        arm_r0 = ring_r5 + core_radius_m / 2
        # arm_r1 = 7135
        arm_r1 = 6400

        # ============== Core
        tel.seed = 74383209

        args = dict(amps=taylor_win(n_taylor, sll), taper_r_min=0.30)
        # 510m worked ok
        tel.add_tapered_core(224 + 2, 480,
                             AnalyseUnwrapV5._taper_r_profile, **args)
        print('final seed =', tel.layouts['tapered_core']['info']['final_seed'])
        # tel.add_ska1_v5(r_max=500)

        # ============== Rings
        # TODO(BM) rotate every other ring so radii don't align
        np.random.seed(1)
        for i, r in enumerate(ring_radii[0:2]):
            print('xxx', i, r)
            # tel.add_ring(num_per_ring, r,
            #              delta_theta=(360/(num_per_ring * 2)) * (i%2))
            tel.add_ring(num_per_ring, r,
                         delta_theta=np.random.randint(low=0, high=360))
            # tel.add_ring(num_per_ring, r, delta_theta=0)

        # tel.layouts['test_st'] = dict(x=np.array([2e3]), y=np.array([0e3]))
        tel.grid_cell_size_m = tel.station_diameter_m
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel

        # ============= Spiral arms
        # tel.add_log_spiral_2(25, arm_r0, arm_r1, 0.515, 3, 'inner_arms', 0)
        tel.add_log_spiral_3(25, self.start_inner, arm_r0, arm_r1, 0.515,
                             self.num_arms, self.theta0_deg)

        # tel.add_symmetric_log_spiral(25, arm_r0, arm_r1, 0.515, 3,
        #                              'inner_arms', self.delta_theta_deg_inner)

        x, _, _ = tel.get_coords_enu()
        print('total stations =', x.size)

        # Plotting layout
        metrics.analyse_telescope(tel, 0, self.eval_metrics)

        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()
        return tel

    @staticmethod
    def _add_core_01(tel, num_stations=224, core_radius_m=510, sll=-28):
        n_taylor = 10000
        tel.num_trials = 4
        tel.trial_timeout_s = 45.0
        tel.seed = 40011245
        args = dict(amps=taylor_win(n_taylor, sll), taper_r_min=0.0)
        tel.add_tapered_core(num_stations, core_radius_m,
                             AnalyseUnwrapV5._taper_r_profile, **args)
        # print('seed =', tel.seed)
        return tel

    def core(self, name='core'):
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name + '_r08')
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        # tel.add_uniform_core(num_stations, core_radius_m)
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model09a(self, name='v6_09a-v5_rw_rings'):
        """v5 with RandallW 3 ring scheme"""
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        radii = [800, 1250, 1700]
        number = [12, 24, 36]
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        tel.add_ska1_v5(r_min=1750)
        tel.add_ska1_v5(r_max=500)
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model09b(self, name='v6_09b-core01_rw_rings_v5'):
        """new core + RandallW 3 ring scheme + v5 arms"""
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        radii = [800, 1250, 1700]
        number = [12, 24, 36]
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        tel.add_ska1_v5(r_min=1750)
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model09c(self, name='v6_09c-core01_rw_rings_spiral_v5'):
        """new core + RandallW 3 ring scheme + v5 arms"""
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        radii = [800, 1250, 1700]
        number = [12, 24, 36]
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        arm_r0, arm_r1 = 1900, 6400
        tel.add_log_spiral_3(25 + 5, self.start_inner, arm_r0, arm_r1, 0.515,
                             self.num_arms, self.theta0_deg)
        # tel.add_ska1_v5(r_min=6450)
        # tel.add_ska1_v5(r_min=10e3)
        print(name, tel.num_stations())
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model09d(self, name='v6_09d-v5_core_rw_rings_spiral_v5'):
        """v5 core + RandallW 3 ring scheme + our spiral"""
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel.add_ska1_v5(r_max=500)
        radii = [800, 1250, 1700]
        number = [12, 24, 36]
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        arm_r0, arm_r1 = 1900, 6400
        tel.add_log_spiral_3(25 + 5, self.start_inner, arm_r0, arm_r1, 0.515,
                             self.num_arms, self.theta0_deg)
        tel.add_ska1_v5(r_min=6450)
        # tel.add_ska1_v5(r_min=10e3)
        print(name, tel.num_stations())
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model10(self, name='v6_10-core01_2rings_v5'):
        """2 ring scheme v5 after 1.7km"""
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        radii = [900, 1700]
        number = [27, 45]
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        tel.add_ska1_v5(r_min=1750)
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model11(self, name='v6_11-core01_3rings_v5'):
        """altered version of RW 3 ring scheme in an attempt to improve
        uniform uv coverage. v5 after 1.7km
        """
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        # radii = [800, 1250, 1700]  # RW values
        radii = [900, 1200, 1700]
        number = [12+5, 24+1, 36-6]
        assert(int(np.sum(number)) == 72)
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        tel.add_ska1_v5(r_min=1750)
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model12(self, name='v6_12-core01_3rings_spiral_v5'):
        """altered version of RW 3 ring scheme in an attempt to improve
        uniform uv coverage. v5 after 1.7km
        """
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        # radii = [800, 1250, 1700]  # RW values
        radii = [900, 1200, 1700]
        number = [12 + 5, 24 + 1, 36 - 6]
        assert (int(np.sum(number)) == 72)
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        tel.add_ska1_v5(r_min=6450)

        # ============= Spiral arms
        arm_r0, arm_r1 = 1800, 6400
        tel.add_log_spiral_3(25, self.start_inner, arm_r0, arm_r1, 0.515,
                             self.num_arms, self.theta0_deg)
        print(tel.num_stations())
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model13(self, name='v6_13-core01_3rings_spiral_v5'):
        """altered version of RW 3 ring scheme in an attempt to improve
        uniform uv coverage. v5 after 1.7km
        """
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        # radii = [800, 1250, 1700]  # RW values
        radii = [900, 1200, 1700]
        # number = [12 + 5, 24 + 1, 36 - 6]
        number = [12 + 5 - 1 + 2, 24 + 1 - 2, 36 - 6 - 3 - 2]
        assert (int(np.sum(number)) == 72 - 6)
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        tel.add_ska1_v5(r_min=6450)

        # ============= Spiral arms
        arm_r0, arm_r1 = 1800, 6400
        tel.add_log_spiral_3(25 + 2 + 5, self.start_inner, arm_r0, arm_r1, 0.515,
                             self.num_arms, self.theta0_deg)
        assert(tel.num_stations() == 512)
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model14(self, name='v6_14-core01_3rings_spiral_v5'):
        """altered version of RW 3 ring scheme in an attempt to improve
        uniform uv coverage. v5 after 1.7km
        """
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        # tel.add_ska1_v5(r_max=500)
        # radii = [800, 1250, 1700]  # RW values
        radii = [750, 1200, 1700]
        number = [25, 27, 17]
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        # tel.add_ska1_v5(r_min=6450)

        # ============= Spiral arms
        arm_r0, arm_r1 = 1800, 6400
        tel.add_log_spiral_3(25 + 6, self.start_inner, arm_r0, arm_r1,
                             0.515, self.num_arms, self.theta0_deg)
        # assert(tel.num_stations() == 512)
        print(tel.name, tel.num_stations())
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics

    def model15(self, name='v6_15-core01_3rings_spiral_v5'):
        """altered version of RW 3 ring scheme in an attempt to improve
        uniform uv coverage. v5 after 1.7km
        """
        metrics = Metrics(self.out_dir)
        tel = SKA1_low_analysis(name)
        tel.station_diameter_m = 35
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg
        tel = self._add_core_01(tel)
        # tel.add_ska1_v5(r_max=500)
        # radii = [800, 1250, 1700]  # RW values
        radii = [750, 1200, 1700]
        number = [21, 27, 21]
        for rn in zip(radii, number):
            tel.add_ring(rn[1], rn[0], 0)
        tel.add_ska1_v5(r_min=6450)

        # ============= Spiral arms
        arm_r0, arm_r1 = 1900, 6400
        tel.add_log_spiral_3(25 + 6, self.start_inner, arm_r0, arm_r1,
                             0.515, self.num_arms, self.theta0_deg)
        # assert(tel.num_stations() == 512)
        print(tel.name, tel.num_stations())
        metrics.analyse_telescope(tel, 0, self.eval_metrics)
        return tel, metrics


if __name__ == '__main__':
    # ====== Options =========================
    snapshot = True
    root = 'TEMP_results_v6'
    remove_existing_results = False
    r_values = [8]  # List of radii to unpack to (or None for all)
    b_max_hist = 10e3
    enable_metrics = dict(
        layout_plot=dict(show_decorations=False, xy_lims=[2e3, 7.5e3, 20e3],
                         plot_radii=[500, 800, 1250, 1.7e3, 6.4e3]),
        uv_grid=False,  # dict(xy_lim=[2e3, 4e3, 10e3], grid_cellsize_m=35/2),
        uv_scatter=False,  # dict(alpha=0.1),
        layout_hist_lin=False,  # dict(b_min=0, b_max=500, num_bins=30),
        uv_hist_lin=False,  # dict(b_min=0, b_max=b_max_hist, num_bins=400,
                           # make_plot=False),
        uv_hist_log=False,  # dict(b_min=10, b_max=2.5e3, num_bins=100),
        psf=False,
        layout_enu=True,
        layout_iantconfig=True
    )
    # ========================================
    if snapshot:
        obs_length_h = 0
        num_times = 1
    else:
        obs_length_h = 4
        num_times = (obs_length_h * 3600) // 60
    out_dir = b'%s_%.1fh' % (root, obs_length_h)

    pprint(enable_metrics)

    results = list()
    unwrap_v5 = AnalyseUnwrapV5(
        out_dir=out_dir, remove_existing_results=remove_existing_results,
        obs_length_h=obs_length_h, num_times=num_times,
        eval_metrics=enable_metrics)
    results.append(analyse_v5(out_dir=out_dir, obs_length_h=obs_length_h,
                              num_times=num_times, eval_metrics=enable_metrics))
    # results.append(unwrap_v5.model03(add_core=True, r_values=r_values))
    # tel_07 = unwrap_v5.model07(add_core=True)
    # tel_08 = unwrap_v5.model08(add_core=True)
    # results.append(unwrap_v5.model09a())
    # results.append(unwrap_v5.model09b())
    # results.append(unwrap_v5.model09c())
    results.append(unwrap_v5.model09d())
    # results.append(unwrap_v5.model10())
    # results.append(unwrap_v5.model11())
    # results.append(unwrap_v5.model12())
    # results.append(unwrap_v5.model13())
    # results.append(unwrap_v5.model14())
    results.append(unwrap_v5.model15())
    # _, _ = unwrap_v5.core()

    # for i, result in enumerate(results):
    #     print(i, result.num_stations())



    _psf_compare = False
    _hist_compare = False
    import matplotlib.pyplot as plt
    if _hist_compare:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.98)
        _key = 'lin_%05.1fkm' % (b_max_hist / 1e3)
        for i, result in enumerate(results):
            tel_name = result[0].name
            _hist = result[1].uv_hist[tel_name][_key]
            x = _hist['hist_x']
            y = _hist['hist_n']
            ax.plot(x, y, label=tel_name)
        for x_ in [800, 1250, 1700]:
            ax.plot([x_, x_], ax.get_ylim(), 'k--')
        for x_ in [900, 1300, 1700]:
            ax.plot([x_, x_], ax.get_ylim(), 'g:')
        ax.legend(loc='best')
        ax.set_xlabel('Baseline length (m)')
        ax.set_ylabel('Baselines per bin')
        ax.grid()
        fig.savefig(join(out_dir, 'AAA_compare_hist.png'))
        plt.close(fig)

    psf_min = 1e-4
    psf_max = 0.5
    if _psf_compare:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.98)
        for i, result in enumerate(results):
            tel_name = result[0].name
            _psf = result[1].psf[tel_name]
            print(_psf.keys())
            # print(_psf['fov'])
            x = _psf['1d_r']
            y = _psf['1d_abs_mean']
            ax.plot(x, y, label=tel_name)
        ax.legend(loc='best')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid()
        ax.set_ylim(psf_min, psf_max)
        ax.set_yscale('log')
        fig.savefig(join(out_dir, 'AAA_compare_psf_abs_mean.png'))
        plt.close(fig)

        if _psf_compare:
            fig, ax = plt.subplots(figsize=(14, 6))
            fig.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.98)
            for i, result in enumerate(results):
                tel_name = result[0].name
                _psf = result[1].psf[tel_name]
                x = _psf['1d_r']
                y = _psf['1d_abs_max']
                ax.plot(x, y, label=tel_name)
            ax.plot([0.007, 0.007], ax.get_ylim(), 'k--')
            ax.plot([0.006, 0.006], ax.get_ylim(), 'k--')
            ax.plot(ax.get_xlim(), [0.005, 0.005], 'k--')
            ax.legend(loc='best')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid()
            ax.set_ylim(psf_min, psf_max)
            ax.set_yscale('log')
            fig.savefig(join(out_dir, 'AAA_compare_psf_abs_max.png'))
            plt.close(fig)

            if _psf_compare:
                fig, ax = plt.subplots(figsize=(14, 6))
                fig.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.98)
                for i, result in enumerate(results):
                    tel_name = result[0].name
                    _psf = result[1].psf[tel_name]
                    x = _psf['1d_r']
                    y = _psf['1d_rms']
                    ax.plot(x, y, label=tel_name)
                ax.legend(loc='best')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.grid()
                ax.set_ylim(psf_min, psf_max)
                ax.set_yscale('log')
                fig.savefig(join(out_dir, 'AAA_compare_psf_rms.png'))
                plt.close(fig)

            if _psf_compare:
                fig, ax = plt.subplots(figsize=(14, 6))
                fig.subplots_adjust(left=0.05, right=0.98, bottom=0.1,
                                    top=0.98)
                for i, result in enumerate(results):
                    tel_name = result[0].name
                    _psf = result[1].psf[tel_name]
                    x = _psf['1d_r']
                    y = _psf['1d_max']
                    ax.plot(x, y, label=tel_name)
                ax.legend(loc='best')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.grid()
                ax.set_ylim(1e-6, psf_max)
                ax.set_yscale('log')
                fig.savefig(join(out_dir, 'AAA_compare_psf_max.png'))
                plt.close(fig)

                # coords_v6 = np.loadtxt()

    # import matplotlib.pyplot as plt
    # x, y, _ = tel_v5.get_coords_enu()
    # fig, ax = plt.subplots(figsize=(8, 8))
    # plt.plot(x, y, 'r+')
    # x, y, _ = tel_v6.get_coords_enu()
    # for xy in zip(x, y):
    #     ax.add_artist(plt.Circle(xy, 35/2, fill=False, color='k'))
    # plt.show()
    # plt.close(fig)

    # Metrics.compare_cum_hist(join(out_dir), log_axis=False)
    # Metrics.compare_hist(join(out_dir), log_axis=True)
    # # Metrics.compare_cum_hist(join(out_dir), log_axis=False)
    # Metrics.compare_psf_1d(join(out_dir))
    # Metrics.compare_hist_2('TEMP_results')
