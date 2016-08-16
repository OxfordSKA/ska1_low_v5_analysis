"""Script for radial unwrapping of v5 clusters."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
from utilities.telescope import Telescope
from utilities.analysis import SKA1_low_analysis
import numpy as np
from os import makedirs
from os.path import isdir, join
from eval_metrics import AnalyseMetrics


def main():
    # -------------- Options --------------------------------------------------
    out_dir = 'TEMP_results'
    name = 'model03'  # Name of the telescope model (prefix in file names)
    # -------------------------------------------------------------------------

    # Generate new telescopes based on v5 by expanding each station cluster.
    core_radius = 500
    outer_radius = 6400
    cluster_radius = 90
    stations_per_cluster = 6
    b = 0.515
    theta0_deg = -48
    start_inner = 417.82
    num_arms = 3
    d_theta = 360 / num_arms

    # Get cluster radii. TODO(BM) incorporate with load of v5 (add to metadata?)
    cluster_x, cluster_y, arm_index = \
        Telescope.cluster_centres_ska_v5(0, outer_radius)
    cluster_r = (cluster_x**2 + cluster_y**2)**0.5
    # Get every 3rd radius as clusters are in 3 arms of common radius.
    cluster_r = cluster_r[::3]

    # Get theta separation between cluster rings for the inner and outer
    # log spirals. TODO(BM) incorporate with load of v5 (add to metadata?)
    delta_theta_deg_inner = Telescope.delta_theta(
        cluster_x[0 * num_arms], cluster_y[0 * num_arms],
        cluster_x[1 * num_arms], cluster_y[1 * num_arms],
        start_inner, b)
    delta_theta_deg_outer = Telescope.delta_theta(
        cluster_x[4 * num_arms], cluster_y[4 * num_arms],
        cluster_x[5 * num_arms], cluster_y[5 * num_arms],
        start_inner, b)
    delta_theta_deg_inner *= (5 / 12)
    delta_theta_deg_outer *= (5 / 12)

    metrics = AnalyseMetrics(out_dir)

    # Loop over cluster radii.
    for i in range(len(cluster_r)):
        print('--- unpacking cluster ring %i ---' % i)
        # Create the telescope model and add the core (from v5)
        tel = SKA1_low_analysis('%s_r%02i' % (name, i))
        tel.add_ska1_v5(r_max=core_radius)

        # Add the SKA1 v5 clusters we are not replacing
        if i < cluster_r.size - 1:
            tel.add_ska1_v5(cluster_r[i + 1] - cluster_radius, outer_radius)

        # Loop over cluster radii
        for j in range(i + 1):
            delta_theta_deg = delta_theta_deg_inner if j <= 3 else \
                delta_theta_deg_outer
            for k in range(num_arms):
                idx = num_arms * j + k
                if j >= 5:
                    tel.add_log_spiral_section(
                        stations_per_cluster, start_inner,
                        cluster_x[idx], cluster_y[idx],
                        b, delta_theta_deg,
                        theta0_deg + arm_index[idx] * d_theta)
                else:
                    tel.add_circular_arc(stations_per_cluster, cluster_x[idx],
                                         cluster_y[idx], d_theta)
        metrics.analyse(tel, cluster_r[i])

    metrics.save(name)


if __name__ == '__main__':
    main()
