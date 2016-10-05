"""Script for unwrapping of v5 clusters."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import os
import shutil
import numpy as np
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
    name = 'ska1_v5'  # Name of the telescope model (prefix in file names)
    outer_radius = 6400 + 90
    # -------------------------------------------------------------------------
    if not isdir(out_dir):
        makedirs(out_dir)

    # Current SKA1 V5 design.
    tel = SKA1_low_analysis(name)
    tel.add_ska1_v5(r_max=outer_radius)
    tel.obs_length_h = obs_length_h
    tel.num_times = num_times
    tel.dec_deg = tel.lat_deg

    metrics = Metrics(out_dir)
    metrics.analyse_telescope(tel, 0.0, eval_metrics)
    metrics.save_results(name)
    return tel


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

    def __add_v5_core_clusters(self, name, i, add_core=True):
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

    def model01(self, name='model01', add_core=True, r_values=None):
        """Model01 == replace clusters with spiral arms"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)

        # Loop over telescope models (replacing additional rings of clusters)
        for i in range(self.cluster_r.size):
            if r_values is not None and i not in r_values:
                continue
            # Create the telescope model (v5 core & clusters not being replaced)
            tel = self.__add_v5_core_clusters(name, i, add_core)

            # Add spiral sections for clusters we are replacing
            for j in range(i + 1):
                # Spiral offset from centre in theta space.
                delta_theta_deg = self.delta_theta_deg_inner if j <= 3 else \
                    self.delta_theta_deg_outer
                for k in range(self.num_arms):
                    idx = self.num_arms * j + k
                    tel.add_log_spiral_section(
                        self.stations_per_cluster, self.start_inner,
                        self.cluster_x[idx], self.cluster_y[idx], self.b,
                        delta_theta_deg,
                        self.theta0_deg + self.arm_index[idx] * self.d_theta)

            # Produce analysis metrics for the telescope
            metrics.analyse_telescope(tel, self.cluster_r[i], self.eval_metrics)

        # Save various comparision metrics.
        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

    def model02(self, name='model02', add_core=True, r_values=None):
        """Model02 == replace clusters with radial arcs"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)

        # Loop over cluster radii
        for i in range(self.cluster_r.size):
            if r_values is not None and i not in r_values:
                continue
            # Create the telescope model (v5 core & clusters not being replaced)
            tel = self.__add_v5_core_clusters(name, i, add_core)

            # Add circular sections for clusters we are replacing
            for j in range(i + 1):
                for k in range(self.num_arms):
                    idx = self.num_arms * j + k
                    tel.add_circular_arc(self.stations_per_cluster,
                                         self.cluster_x[idx],
                                         self.cluster_y[idx], self.d_theta)
            metrics.analyse_telescope(tel, self.cluster_r[i],
                                      self.eval_metrics)

        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

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
            tel = self.__add_v5_core_clusters(name, i, add_core)

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
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

    def model04(self, name='model04', add_core=True, r_values=None):
        """Model04 == replace clusters with circles then perturbed circles"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)
        r0 = 500
        r1 = 6400
        perturb_r0 = 300
        perturb_r1 = 0.8 * (self.cluster_r[-1] - self.cluster_r[-2])
        # print('perturb_r1', perturb_r1)
        # Loop over cluster radii
        for i in range(self.cluster_r.size):
            if r_values is not None and i not in r_values:
                continue
            np.random.seed(5)
            # Create the telescope model (v5 core & clusters not being replaced)
            tel = self.__add_v5_core_clusters(name, i, add_core)

            # Loop over cluster radii we are replacing
            for j in range(i + 1):
                for k in range(self.num_arms):
                    idx = self.num_arms * j + k
                    if j >= 5:
                        tel.add_circular_arc_perturbed(
                            self.stations_per_cluster,
                            self.cluster_x[idx],
                            self.cluster_y[idx], self.d_theta,
                            r0, r1, perturb_r0, perturb_r1)
                    else:
                        tel.add_circular_arc(
                            self.stations_per_cluster,
                            self.cluster_x[idx],
                            self.cluster_y[idx], self.d_theta)
            metrics.analyse_telescope(tel, self.cluster_r[i],
                                      self.eval_metrics)

        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

    def model05(self, name='model05', add_core=True, r_values=None):
        """Model05 == random radial profile"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)

        # Loop over cluster radii
        for i in range(self.cluster_r.size):
            if r_values is not None and i not in r_values:
                continue
            # Create the telescope model (v5 core & clusters not being replaced)
            tel = self.__add_v5_core_clusters(name, i, add_core)

            # Add random radial profile.
            c_x = self.cluster_x[0:self.num_arms * (i + 1)]
            c_y = self.cluster_y[0:self.num_arms * (i + 1)]
            tel.add_random_profile(self.num_arms * self.cluster_r.size * 6,
                                   c_x, c_y,
                                   self.core_radius,
                                   self.outer_radius,
                                   self.num_arms * (i + 1) * 6)
            metrics.analyse_telescope(tel, self.cluster_r[i],
                                      self.eval_metrics)

        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

    def model06(self, name='model06', add_core=True, r_values=None):
        """Model06 == no cables > 1km, an improved core"""
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


        # # Loop over telescope models (replacing additional rings of clusters)
        # for i in range(self.cluster_r.size):
        #     if r_values is not None and i not in r_values:
        #         continue
        #     # Create the telescope model (v5 core & clusters not being replaced)
        #     tel = self.__add_v5_core_clusters(name, i, add_core)
        #
        #     # Add spiral sections for clusters we are replacing
        #     for j in range(i + 1):
        #         # Spiral offset from centre in theta space.
        #         delta_theta_deg = self.delta_theta_deg_inner if j <= 3 else \
        #             self.delta_theta_deg_outer
        #         for k in range(self.num_arms):
        #             idx = self.num_arms * j + k
        #             tel.add_log_spiral_section(
        #                 self.stations_per_cluster, self.start_inner,
        #                 self.cluster_x[idx], self.cluster_y[idx], self.b,
        #                 delta_theta_deg,
        #                 self.theta0_deg + self.arm_index[idx] * self.d_theta)
        #
        #     # Produce analysis metrics for the telescope
        #     metrics.analyse_telescope(tel, self.cluster_r[i], self.eval_metrics)
        #
        # # Save various comparision metrics.
        # metrics.save_results(name)
        # # metrics.plot_cable_length_compare()
        # metrics.plot_comparisons()

    def model07(self, name='model07', add_core=True):
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

        def taper_r_profile(r, amps, taper_r_min=0):
            """Nearest neighbour matching. FIXME(BM) double check this"""
            r = np.asarray(r)
            idx = np.round(((r - taper_r_min) / (1 - taper_r_min)) * (amps.size - 1))
            values = np.asarray(amps[idx.astype(np.int)])
            values[r < taper_r_min] = 1.0
            return values

        tel = SKA1_low_analysis(name + '_r08')
        tel.station_diameter_m = 40
        tel.obs_length_h = self.obs_length_h
        tel.num_times = self.num_times
        tel.dec_deg = tel.lat_deg

        # Core
        core_radius_m = 480
        sll = -24
        n_taylor = 10000
        tel.num_trials = 10
        tel.trial_timeout_s = 30.0
        tel.seed = 24183655

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
        arm_r0 = 1805
        arm_r0 = ring_r5 + core_radius_m / 2
        arm_r1 = 7135
        arm_r1 = 6400

        # ============== Core

        # args = dict(amps=taylor_win(n_taylor, sll), taper_r_min=0.50)
        # tel.add_tapered_core(224 + 6, core_radius_m,
        #                      taper_r_profile, **args)
        # print('final seed =', tel.layouts['tapered_core']['info']['final_seed'])
        # tel.add_ska1_v5(r_max=500)
        tel.add_ska1_v5(r_max=500)

        # ============== Rings
        # TODO(BM) rotate every other ring so radii don't align
        for i, r in enumerate(ring_radii):
            # tel.add_ring(num_per_ring, r, delta_theta=(360/(num_per_ring * 2)) * (i%2))
            tel.add_ring(num_per_ring, r, delta_theta=np.random.randint(low=0, high=360))
            # tel.add_ring(num_per_ring, r, delta_theta=0)

        # ============= Spiral arms
        tel.add_log_spiral_2(25, arm_r0, arm_r1, 0.515, 3, 'inner_arms', 0)
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


if __name__ == '__main__':
    # ====== Options =========================
    snapshot = True
    out_dir = b'TEMP_results_0h' if snapshot else b'TEMP_results_4h'
    remove_existing_results = False
    r_values = [8]  # List of radii to unpack to (or None for all)
    enable_metrics = dict(
        #layout_plot=dict(xy_lim=50e3, plot_radii=[0.5e3, 6.4e3, 40e3]),
        # layout_plot=True,
        layout_plot=dict(show_decorations=True, xy_lim=1.7e3, plot_radii=[500, 1.7e3, 6.4e3]),
        layout_matlab=False,
        layout_pickle=False,
        layout_enu=False,
        layout_iantconfig=False,
        cable_length_1=False,
        cable_length_2=False,
        cable_length_3=False,
        uv_grid=True,
        uv_hist=True,
        mst_network=False,
        psf_rms=False,
        uv_gap=False,
        psf=True,
    )
    # ========================================
    if snapshot:
        obs_length_h = 0
        num_times = 1
    else:
        obs_length_h = 4
        num_times = (obs_length_h * 3600) // 60

    pprint(enable_metrics)

    unwrap_v5 = AnalyseUnwrapV5(
        out_dir=out_dir, remove_existing_results=remove_existing_results,
        obs_length_h=obs_length_h, num_times=num_times,
        eval_metrics=enable_metrics)
    # unwrap_v5.model01(add_core=True, r_values=r_values)
    # unwrap_v5.model02(add_core=True, r_values=r_values)
    # unwrap_v5.model03(add_core=True, r_values=r_values)
    # unwrap_v5.model04(add_core=True, r_values=r_values)
    # unwrap_v5.model05(add_core=True, r_values=r_values)
    # unwrap_v5.model06(add_core=True, r_values=r_values)
    tel_v6 = unwrap_v5.model07(add_core=True)
    tel_v5 = analyse_v5(out_dir=out_dir, obs_length_h=obs_length_h,
                        num_times=num_times, eval_metrics=enable_metrics)

    import matplotlib.pyplot as plt

    x, y, _ = tel_v5.get_coords_enu()
    fig, ax = plt.subplots(figsize=(8, 8))
    for xy in zip(x, y):
        ax.add_artist(plt.Circle(xy, 35/2, filled=False, color='k'))
    x, y, _ = tel_v6.get_coords_enu()

    plt.show()
    plt.close(fig)


    # Metrics.compare_cum_hist(join(out_dir), log_axis=False)
    Metrics.compare_hist(join(out_dir), log_axis=True)
    # Metrics.compare_cum_hist(join(out_dir), log_axis=False)
    Metrics.compare_psf_1d(join(out_dir))
    # Metrics.compare_hist_2('TEMP_results')
