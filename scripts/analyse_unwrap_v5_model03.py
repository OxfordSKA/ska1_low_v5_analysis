"""Script for radial unwrapping of v5 clusters."""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
from utilities.telescope import Telescope
from utilities.analysis import SKA1_low_analysis
import numpy as np
from os import makedirs
from os.path import isdir, join


def main():
    # -------------- Options --------------------------------------------------
    out_dir = 'results'
    name = 'model03'  # Name of the telescope model (prefix in file names)
    # -------------------------------------------------------------------------

    if not isdir(out_dir):
        makedirs(out_dir)

    # Generate new telescopes based on v5 by expanding each station cluster.
    r_cut = 6400
    b = 0.515
    theta0_deg = -48
    start_inner = 417.82
    num_arms = 3
    d_theta = 360 / num_arms

    # Get cluster radii.
    cluster_x, cluster_y, arm_index = Telescope.cluster_centres_ska_v5(0, r_cut)
    cluster_r = (cluster_x**2 + cluster_y**2)**0.5
    cluster_r = cluster_r[::3]  # Get every 3rd radius.

    # Get theta separation between cluster rings for the inner and outer
    # log spirals.
    delta_theta_deg_inner = Telescope.delta_theta(
        cluster_x[0 * num_arms], cluster_y[0 * num_arms],
        cluster_x[1 * num_arms], cluster_y[1 * num_arms],
        start_inner, b)
    delta_theta_deg_outer = Telescope.delta_theta(
        cluster_x[4 * num_arms], cluster_y[4 * num_arms],
        cluster_x[5 * num_arms], cluster_y[ 5 *num_arms],
        start_inner, b)
    delta_theta_deg_inner *= (5 / 12)
    delta_theta_deg_outer *= (5 / 12)

    cable_length = np.zeros(len(cluster_r))

    # Loop over cluster radii.
    for i in range(len(cluster_r)):
        # if i != 3:
        #     continue
        print('--- unpacking cluster ring %i ---' % i)
        # Create the telescope and add the core.
        tel1 = SKA1_low_analysis()
        tel1.add_ska1_v5(None, 500)

        # Add SKA1 V5 clusters from this radius outwards.
        if i < len(cluster_r) - 1:
            r = cluster_r[i + 1]
            tel1.add_ska1_v5(r - 90, 6400)

        # Add spiral sections up to this radius.
        for j in range(i + 1):
            delta_theta_deg = delta_theta_deg_inner if j <= 3 else \
                delta_theta_deg_outer
            for k in range(num_arms):
                idx = num_arms * j + k
                if j >= 5:
                    tel1.add_log_spiral_section(
                        6, start_inner,
                        cluster_x[idx], cluster_y[idx],
                        b, delta_theta_deg,
                        theta0_deg + arm_index[idx] * d_theta)
                    # # theta_inc = ((d_theta - 30) / 4)
                    # # angle = d_theta - theta_inc * (j - 4)
                    # # print('... ', j, angle, theta_inc)
                    # # angle = d_theta * (cluster_r[5] / cluster_r[j])**3
                    # tel1.add_circular_arc(
                    #     6, cluster_x[idx], cluster_y[idx], d_theta)
                else:
                    tel1.add_circular_arc(
                        6, cluster_x[idx], cluster_y[idx], d_theta )

        # tel1.save_iantconfig(join(out_dir, '%s_r%02i' % (name, i)))
        # tel1.plot_layout(color='k', show_decorations=True,
        #                  filename=join(out_dir, '%s_r%02i_layout.png' % (name, i)),
        #                  plot_r=7000)
        tel1.plot_grid(join(out_dir, '%s_r%02i_uv_grid.png' % (name, i)),
                       x_lim=[-r_cut * 2, r_cut * 2],
                       y_lim=[-r_cut * 2, r_cut * 2], plot_radii=[r_cut * 2])

        # tel1.plot_grid(filename=join(out_dir, 'uv_grid_%02i.png' % i))
        # # tel1.eval_psf_rms_r()
        # # tel1.plot_network()
        # print(i, cluster_r[i])
        # cable_length[i] = tel1.cable_length(
        #     plot=True, plot_filename=join(out_dir, 'cable_length_%02i.png' % i))

    # np.save(join(out_dir, 'cable_length.npz'),
    #         cluster_=cluster_r, cable_length=cable_length)

    # print()
    # print(ref_cable_length)
    # print(cable_length)
    # delta_cable_length = cable_length - ref_cable_length
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.plot(cluster_r, delta_cable_length)
    # ax.set_ylabel('Cable length increase from v5 reference (m)')
    # ax.set_xlabel('unpacking radius (m)')
    # ax.set_yscale('log')
    # fig.savefig(join(out_dir, 'cable_length.png'))
    # plt.show()

if __name__ == '__main__':
    main()
