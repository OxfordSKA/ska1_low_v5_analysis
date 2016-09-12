"""Script for unwrapping of v5 clusters."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import os
import shutil
from utilities.telescope import Telescope
from utilities.analysis import SKA1_low_analysis
from utilities.eval_metrics import Metrics
from os.path import join, isdir
from os import makedirs


def analyse_v5(out_dir='TEMP_results'):
    """Generate and analyse reference v5 layout."""
    # -------------- Options --------------------------------------------------
    name = 'ska1_v5'  # Name of the telescope model (prefix in file names)
    outer_radius = 6400
    # -------------------------------------------------------------------------
    if not isdir(out_dir):
        makedirs(out_dir)

    # Current SKA1 V5 design.
    tel = SKA1_low_analysis(name)
    tel.add_ska1_v5(r_max=outer_radius)
    metrics = Metrics(out_dir)
    metrics.analyse_telescope(tel, 0.0)
    metrics.save_results(name)


class AnalyseUnwrapV5(object):
    """Class to unwrap v5 clusters and analyse the result."""
    def __init__(self, out_dir='TEMP_results', remove_existing_results=True):
        self.out_dir = out_dir

        if remove_existing_results and os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

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
        print('-- Telescope: %s' % tel.name)
        if add_core:
            tel.add_ska1_v5(r_max=self.core_radius)

        # Add the SKA1 v5 clusters we are not replacing
        if i < self.cluster_r.size - 1:
            tel.add_ska1_v5(self.cluster_r[i + 1] - self.cluster_radius,
                            self.outer_radius + self.cluster_radius)
        return tel

    def model01(self, name='model01', add_core=True):
        """Model01 == replace clusters with spiral arms"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)

        # Loop over telescope models (replacing additional rings of clusters)
        for i in range(self.cluster_r.size):
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
            metrics.analyse_telescope(tel, self.cluster_r[i])

        # Save various comparision metrics.
        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

    def model02(self, name='model02', add_core=True):
        """Model02 == replace clusters with radial arcs"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)

        # Loop over cluster radii
        for i in range(self.cluster_r.size):
            # Create the telescope model (v5 core & clusters not being replaced)
            tel = self.__add_v5_core_clusters(name, i, add_core)

            # Add circular sections for clusters we are replacing
            for j in range(i + 1):
                for k in range(self.num_arms):
                    idx = self.num_arms * j + k
                    tel.add_circular_arc(self.stations_per_cluster,
                                         self.cluster_x[idx],
                                         self.cluster_y[idx], self.d_theta)
            metrics.analyse_telescope(tel, self.cluster_r[i])

        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

    def model03(self, name='model03', add_core=True):
        """Model03 == replace clusters with radial arcs and log spirals"""
        # Initialise metrics class
        metrics = Metrics(self.out_dir)

        # Loop over cluster radii
        for i in range(self.cluster_r.size):
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
            metrics.analyse_telescope(tel, self.cluster_r[i])

        metrics.save_results(name)
        # metrics.plot_cable_length_compare()
        metrics.plot_comparisons()

    def model04(self, name='model04', add_core=True):
        """Model04 == replace clusters with circles then perturbed circles"""
        #
        metrics = Metrics(self.out_dir)

        pass

def main():
    unwrap_v5 = AnalyseUnwrapV5(remove_existing_results=True)
    unwrap_v5.model01(add_core=True)
    unwrap_v5.model02(add_core=True)
    unwrap_v5.model03(add_core=True)
    analyse_v5()

if __name__ == '__main__':
    # main()
    Metrics.compare_cum_hist(join('TEMP_results'), log_axis=False)
    Metrics.compare_hist(join('TEMP_results'), log_axis=False)
