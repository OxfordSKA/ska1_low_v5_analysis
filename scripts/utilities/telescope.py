# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from math import (log, log10, radians, cos, sin, pi, acosh, atan2, exp,
                  degrees, sqrt, ceil)
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle
import numpy as np
from .layout import Layout


class Telescope(object):
    def __init__(self, name=''):
        self.name = name
        self.lon_deg = 0
        self.lat_deg = 0
        self.alt_m = 0
        self.station_diameter_m = 35
        self.trial_timeout_s = 2.0  # s
        self.num_trials = 5
        self.verbose = False
        self.seed = None
        self.layouts = dict()
        self.clusters = {'cx': np.array([]), 'cy': np.array([])}

    def clear_layouts(self):
        self.layouts.clear()

    def save_enu(self, filename):
        """Save the telescope model as ascii CSV file."""
        x, y, z = self.get_coords_enu()
        coords = np.vstack([x, y, z]).T
        #np.savetxt(filename, coords, fmt=b'%.12e %.12e %.12e')
        np.savetxt(filename, coords, fmt=b'%.12e')

    def save_iantconfig(self, filename_root):
        x, y, _ = self.get_coords_enu()
        d = np.ones_like(x)  # set to 1 to avoid shadowing in iantconfig
        coords = np.vstack([d, x, y]).T
        coords = np.vstack([coords, [0, self.lon_deg, self.lat_deg]])
        np.savetxt('%s.enu.%ix%i.txt' % (filename_root, x.size, 3),
                   coords, fmt=b'%-5i %.12f %.12f')

    def to_oskar_telescope_model(self, filename):
        pass

    def add_uniform_core(self, num_stations, r_max_m, r_min_m=0):
        """Add uniform random core"""
        if self.seed is None:
            self.seed = np.random.randint(1, 1e8)
        layout = Layout(self.seed, self.trial_timeout_s, self.num_trials)
        layout.uniform_cluster(num_stations, self.station_diameter_m,
                               r_max_m, r_min_m)
        self.layouts['uniform_core'] = dict(x=layout.x, y=layout.y)

    def add_tapered_core(self, num_stations, r_max_m, taper_func, **kwargs):
        """Add a tapered core"""
        if self.seed is None:
            self.seed = np.random.randint(1, 1e8)
        try:
            layout_ = Layout.rand_tapered_2d_trials(
                num_stations, r_max=r_max_m, r_min=0.0,
                min_sep=self.station_diameter_m,
                trial_timeout=self.trial_timeout_s, num_trials=self.num_trials,
                seed0=self.seed, taper_func=taper_func, **kwargs)
            self.layouts['tapered_core'] = dict(
                x=layout_.x, y=layout_.y, info=layout_.info,
                taper_func=taper_func, r_max_m=r_max_m, kwargs=kwargs)
            self.seed = layout_.info['final_seed']
        except RuntimeError as e:
            print('*** ERROR ***:', e.message)


    def add_hex_core(self, r_max_m, theta0_deg=0.0):
        """Add hexagonal lattice to the core"""
        layout = Layout()
        layout.hex_lattice(self.station_diameter_m, r_max_m, theta0_deg)
        self.layouts['hex_core'] = dict(x=layout.x, y=layout.y)

    @staticmethod
    def cluster_centres_ska_v5(r_min=None, r_max=None):
        """Generate cluster centres for SKA1 V5 between given radii."""
        # Spiral parameters for inner and outer regions.
        num_arms = 3
        num_per_arm = 5
        start_inner = 417.82
        end_inner = 1572.13
        b_inner = 0.513
        theta0_inner = -48
        start_outer = 2146.78
        end_outer = 6370.13
        b_outer = 0.52
        theta0_outer = 135
        x_inner, y_inner = Telescope.symmetric_log_spiral(
            num_per_arm, start_inner, end_inner, b_inner, num_arms,
            theta0_inner)
        x_outer, y_outer = Telescope.symmetric_log_spiral(
            num_per_arm, start_outer, end_outer, b_outer, num_arms,
            theta0_outer)
        x = np.concatenate((x_inner, x_outer))
        y = np.concatenate((y_inner, y_outer))
        r = (x**2 + y**2)**0.5
        arm_index = [i // num_per_arm for i in range(num_per_arm * num_arms)]
        arm_index = np.hstack((arm_index, arm_index))

        # Sort by radius and remove the 3 innermost stations.
        idx = r.argsort()
        x = x[idx]
        y = y[idx]
        r = r[idx]
        arm_index = arm_index[idx]
        x, y, r, arm_index = (x[3:], y[3:], r[3:], arm_index[3:])

        if r_min and r_max:
            idx = np.where(np.logical_and(r >= r_min, r <= r_max))
            x, y, arm_index = x[idx], y[idx], arm_index[idx]
        elif r_min:
            idx = np.where(r >= r_min)
            x, y, arm_index = x[idx], y[idx], arm_index[idx]
        elif r_max:
            idx = np.where(r <= r_max)
            x, y, arm_index = x[idx], y[idx], arm_index[idx]
        return x, y, arm_index

    @staticmethod
    def cluster_radii_ska_v5(r_min=None, r_max=None):
        """Return cluster radii for the SKA1 v5 configuration"""
        cluster_x, cluster_y, arm_index = \
            Telescope.cluster_centres_ska_v5(r_min, r_max)
        cluster_r = (cluster_x**2 + cluster_y**2)**0.5
        return cluster_r[::3]

    @staticmethod
    def log_spiral(n, r0, r1, b):
        t_max = log(r1 / r0) / b
        t = np.linspace(0, t_max, n)
        tmp = r0 * np.exp(b * t)
        x = tmp * np.cos(t)
        y = tmp * np.sin(t)
        return x, y

    @staticmethod
    def log_spiral_2(n, r0_ref, r0, r1, b):
        t_max = log(r1 / r0_ref) / b
        t_min = log(r0 / r0_ref) / b
        t_inc = (t_max - t_min) / n
        t = np.linspace(t_min, t_max, n)
        # t = np.arange(n) * t_inc + t_min + (t_inc / 2)
        tmp = r0_ref * np.exp(b * t)
        x = tmp * np.cos(t)
        y = tmp * np.sin(t)
        return x, y

    @staticmethod
    def circular_arc(n, r, delta_theta):
        t_inc = delta_theta / n
        t = np.arange(n) * t_inc - delta_theta / 2 + (t_inc / 2)
        x = r * np.cos(np.radians(t))
        y = r * np.sin(np.radians(t))
        return x, y

    @staticmethod
    def r_range_for_centre(cx, cy, r0_ref, delta_theta_deg, b, theta_offset=0):
        cr = (cx**2 + cy**2)**0.5
        theta = log(cr / r0_ref) / b  # Angle to the centre
        t0 = theta - radians(delta_theta_deg) + radians(theta_offset)
        t1 = theta + radians(delta_theta_deg) + radians(theta_offset)
        r0 = r0_ref * np.exp(b * t0)
        r1 = r0_ref * np.exp(b * t1)
        return r0, r1

    @staticmethod
    def spiral_to_arms(x, y, num_arms, theta0_deg):
        d_theta = 360 / num_arms
        for i in range(num_arms):
            x[i::num_arms], y[i::num_arms] = Layout.rotate_coords(
                x[i::num_arms], y[i::num_arms], theta0_deg + d_theta * i)
        return x, y

    @staticmethod
    def symmetric_log_spiral(n, r0, r1, b, num_arms, theta0_deg):
        x, y = Telescope.log_spiral(n, r0, r1, b)
        d_theta = 360 / num_arms
        x_final = np.zeros(n * num_arms)
        y_final = np.zeros(n * num_arms)
        for arm in range(num_arms):
            x_, y_ = Layout.rotate_coords(
                x, y, theta0_deg + d_theta * arm)
            x_final[arm * n:(arm + 1) * n] = x_
            y_final[arm * n:(arm + 1) * n] = y_
        return x_final, y_final

    @staticmethod
    def delta_theta(cx1, cy1, cx2, cy2, r0_ref, b):
        """cx1, cy1 has to be closer to the origin than cx2, cy2"""
        r0 = (cx1**2 + cy1**2)**0.5
        r1 = (cx2**2 + cy2**2)**0.5
        t_max = log(r1 / r0_ref) / b
        t_min = log(r0 / r0_ref) / b
        return degrees(t_max - t_min)

    def add_log_spiral(self, n, r0, r1, b, num_arms, theta0_deg=0.0):
        """Add spiral arms by rotating a single spiral of n positions"""
        x, y = self.log_spiral(n, r0, r1, b)
        x, y = self.spiral_to_arms(x, y, num_arms, theta0_deg)
        keys = self.layouts.keys()
        self.layouts['spiral_arms' + str(len(keys))] = {'x': x, 'y': y}

    def add_symmetric_log_spiral(self, n, r0, r1, b, num_arms, name,
                                 theta0_deg):
        """Add symmetric spiral arms."""
        x, y = self.symmetric_log_spiral(n, r0, r1, b, num_arms, theta0_deg)
        keys = self.layouts.keys()
        self.layouts[name + str(len(keys))] = {'x': x, 'y': y}

    def add_circular_arc(self, n, cx, cy, delta_theta):
        r = (cx**2 + cy**2)**0.5
        t = degrees(atan2(cy, cx))
        x, y = self.circular_arc(n, r, delta_theta)
        x, y = Layout.rotate_coords(x, y, t)
        keys = self.layouts.keys()
        self.layouts['circular_arc' + str(len(keys))] = {
            'x': x, 'y': y, 'cx': cx, 'cy': cy}

    def add_circular_arc_perturbed(self, n, cx, cy, delta_theta, r0, r1,
                                   perturb_r0, perturb_r1):
        r = (cx**2 + cy**2)**0.5
        t = degrees(atan2(cy, cx))
        x, y = self.circular_arc(n, r, delta_theta)
        x, y = Layout.rotate_coords(x, y, t)
        scale = perturb_r0 + (perturb_r1 - perturb_r0) * ((r - r0) / (r1 - r0))
        for i in range(len(x)):
            t = np.random.rand() * 2 * np.pi
            rt = np.random.rand() * scale
            x[i] += rt * cos(t)
            y[i] += rt * sin(t)
        keys = self.layouts.keys()
        self.layouts['circular_arc' + str(len(keys))] = {
            'x': x, 'y': y, 'cx': cx, 'cy': cy}

    def add_random_profile(self, n, cx, cy, r0, r1, num_selected):
        r = np.logspace(log10(r0), log10(r1), n)

        # Scale profile to maximum radius.
        r = np.sort(r)
        r *= (r1 / np.max(r))

        # Allocate space for modified x,y coordinates.
        x = np.zeros(n)
        y = np.zeros(n)
        np.random.seed(5)

        # Loop over radial positions to generate non-overlapping x,y
        # coordinates.
        for i in range(n):
            trial = 0
            while True:
                # Generate theta (uniform from 0 to 2pi)
                theta = 2.0 * pi * np.random.uniform()
                t_x = r[i] * cos(theta)
                t_y = r[i] * sin(theta)

                # Check distance to all other stations up to this one.
                min_dist = 1e100
                for j in range(i):
                    d_x = x[j] - t_x
                    d_y = y[j] - t_y
                    d = sqrt(d_x * d_x + d_y * d_y)
                    if d < min_dist:
                        min_dist = d

                # If minimum distance is greater than the required minimum
                # separation, store coordinates and go to the next point.
                # Otherwise, keep trying.
                if min_dist >= self.station_diameter_m:
                    x[i] = t_x
                    y[i] = t_y
                    break
                else:
                    trial += 1

                # Check if we've exceeded the maximum number of trials at
                # this radius.
                if trial > 500:
                    r[i] += 1
                    print("  Increasing radius to %.2f m" % (r[i]))

        x = x[0:num_selected]
        y = y[0:num_selected]
        for i in range(len(cx)):
            keys = self.layouts.keys()
            self.layouts['random_profile' + str(len(keys))] = {
                'x': x[i*6:(i+1)*6], 'y': y[i*6:(i+1)*6], 'cx': cx[i], 'cy': cy[i]}

    def add_log_spiral_section(self, n, r0_ref, cx, cy, b, delta_theta,
                               theta0_deg, theta_offset=0):
        r0, r1 = self.r_range_for_centre(cx, cy, r0_ref, delta_theta, b,
                                         theta_offset)
        x, y = self.log_spiral_2(n, r0_ref, r0, r1, b)
        x, y = Layout.rotate_coords(x, y, theta0_deg)
        keys = self.layouts.keys()
        self.layouts['log_spiral_section' + str(len(keys))] = {
            'x': x, 'y': y, 'cx': cx, 'cy': cy, 'r_min': r0, 'r_max': r1}

    def add_log_spiral_clusters(self, num_clusters, num_arms, r0, r1, b,
                                stations_per_cluster, cluster_radius_m,
                                theta0_deg=0.0):
        """Add spiral arm clusters.
        Spiral arm positions generated come from a single single ar
        Note: the random number generator respects class variables
          self.seed
          self.trial_timeout_s
          self.num_trials
        """
        if self.seed is None:
            self.seed = np.random.randint(1, 1e8)
        x_, y_, info = Layout.generate_clusters(
            num_clusters, stations_per_cluster, cluster_radius_m,
            self.station_diameter_m, self.trial_timeout_s, self.num_trials,
            self.seed, r_min=0.0)

        cx, cy = self.log_spiral(num_clusters, r0, r1, b)
        cx, cy = self.spiral_to_arms(cx, cy, num_arms, theta0_deg)
        x = np.zeros(num_clusters * stations_per_cluster)
        y = np.zeros(num_clusters * stations_per_cluster)
        for i in range(num_clusters):
            x[i * stations_per_cluster:(i + 1) * stations_per_cluster] = \
                x_[i] + cx[i]
            y[i * stations_per_cluster:(i + 1) * stations_per_cluster] = \
                y_[i] + cy[i]

        self.layouts['spiral_clusters'] = {'x': x, 'y': y, 'cx': cx, 'cy': cy,
                                           'cr': cluster_radius_m}

    def add_ska1_v5(self, r_min=None, r_max=None):
        """Add SKA1 V5 layout between the given radii."""
        # Load the station coordinates.
        path = os.path.dirname(os.path.abspath(__file__))
        coords = np.loadtxt(join(path, 'data', 'v5_enu.txt'))
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        r = (x**2 + y**2)**0.5

        cluster_radius = 90  # This just seems to work (not confirmed)

        if r_min and r_max:
            idx = np.where(np.logical_and(r >= r_min, r <= r_max))
            x, y, z = x[idx], y[idx], z[idx]
        elif r_min:
            idx = np.where(r >= r_min)
            x, y, z = x[idx], y[idx], z[idx]
        elif r_max:
            idx = np.where(r <= r_max)
            x, y, z = x[idx], y[idx], z[idx]

        # Get the cluster centres within the given range.
        cluster_x, cluster_y, _ = \
            Telescope.cluster_centres_ska_v5(r_min, r_max)

        # Loop over clusters and extract stations within a 90 m radius.
        for cx, cy in zip(cluster_x, cluster_y):
            dr = ((x - cx)**2 + (y - cy)**2)**0.5
            idx = np.where(dr <= cluster_radius)
            tx, ty, tz = x[idx], y[idx], z[idx]
            if tx.size > 0:
                keys = self.layouts.keys()
                self.layouts['ska1_v5_cluster' + str(len(keys))] = {
                    'x': tx, 'y': ty, 'z': tz, 'cx': cx, 'cy': cy,
                    'cr': cluster_radius, 'r_min': r_min, 'r_max': r_max}
                x = np.delete(x, idx)
                y = np.delete(y, idx)
                z = np.delete(z, idx)

        if x.size > 0:
            # Add any remaining stations that were not assigned to a cluster.
            self.layouts['ska1_v5'] = {'x': x, 'y': y, 'z': z,
                                       'r_min': r_min, 'r_max': r_max}

    def num_stations(self):
        if not self.layouts:
            raise RuntimeError('No layout defined!')
        n = 0
        for name in self.layouts:
            layout = self.layouts[name]
            n += layout['x'].shape[0]
        return n

    def get_coords_enu(self):
        if not self.layouts:
            raise RuntimeError('No layout defined!')
        n = self.num_stations()
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        i = 0
        for name in self.layouts:
            layout = self.layouts[name]
            n0 = layout['x'].shape[0]
            x[i:i+n0] = layout['x']
            y[i:i+n0] = layout['y']
            if 'z' in layout:
                z[i:i+n0] = layout['z']
            i += n0
        return x, y, z

    def get_centres_enu(self):
        if not self.layouts:
            raise RuntimeError('No layout defined!')
        n = self.num_stations()
        cx, cy = list(), list()
        i = 0
        for name in self.layouts:
            layout = self.layouts[name]
            if 'cx' in layout and 'cy' in layout:
                cx.append(layout['cx'])
                cy.append(layout['cy'])
        return np.array(cx), np.array(cy)

    def plot_layout(self, filename=None, mpl_ax=None,
                    show_decorations=False, plot_radii=[],
                    xy_lim=None, color='k'):
        plot_nearest = False
        if not self.layouts:
            raise RuntimeError('No layout defined, nothing to plot!')
        if mpl_ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            ax = mpl_ax
        r_max = 0
        for name in self.layouts:
            layout = self.layouts[name]
            x_, y_ = layout['x'], layout['y']
            r = (x_**2 + y_**2)**0.5
            r_max = max(np.max(r), r_max)

            colour = color
            filled = False
            radius = self.station_diameter_m / 2
            for xy in zip(x_, y_):
                c = Circle(xy, radius=radius, fill=filled, color=colour)
                ax.add_artist(c)

            if show_decorations:
                if 'r_min' in layout and layout['r_min'] is not None:
                    ax.add_artist(Circle((0, 0), layout['r_min'], fill=False,
                                         linestyle='--', color='0.5', lw=0.5,
                                         alpha=0.5))

                if 'r_max' in layout and layout['r_max'] is not None:
                    ax.add_artist(Circle((0, 0), layout['r_max'], fill=False,
                                         linestyle='--', color='0.5', lw=0.5,
                                         alpha=0.5))

                # Plot cluster radii, if present
                if 'cx' in layout and 'cy' in layout and 'cr' in layout:
                    for xy in zip([layout['cx']], [layout['cy']]):
                        ax.add_artist(Circle(xy, radius=layout['cr'],
                                             fill=False, color='b',
                                             alpha=0.5))

                # Plot cluster centres, if present
                if 'cx' in layout and 'cy' in layout:
                    for cx, cy in zip([layout['cx']], [layout['cy']]):
                        ax.plot(cx, cy, 'r+', ms=5)

                if 'taper_func' in layout and 'r_max_m' in layout:
                    for k, xy in enumerate(zip(x_, y_)):
                        r_ = (self.station_diameter_m / 2) / \
                             layout['taper_func'](r[k] / layout['r_max_m'],
                                                  **layout['kwargs'])
                        ax.add_artist(Circle(xy, r_, fill=False,
                                                 linestyle='-', color='0.5',
                                                 alpha=0.5))

                if plot_nearest and 'info' in layout and \
                            'attempt_id' in layout['info']:
                    info = layout['info']
                    attempt_id = info['attempt_id']
                    if 'i_min' in info[attempt_id]:
                        i_min = info[attempt_id]['i_min']
                        for k, (x, y) in enumerate(zip(x_, y_)):
                            if i_min[k] < 0:
                                continue
                            dx = x_[i_min[k]] - x
                            dy = y_[i_min[k]] - y
                            ax.arrow(x, y, dx, dy,
                                     head_width=1.5, head_length=3,
                                     overhang=0, length_includes_head=False)

        for r in plot_radii:
            color = r[1] if isinstance(r, tuple) else 'r'
            radius = r[0] if isinstance(r, tuple) else r
            ax.add_artist(plt.Circle((0, 0), radius, fill=False, color=color))

        if xy_lim:
            ax.set_xlim(-xy_lim, xy_lim)
            ax.set_ylim(-xy_lim, xy_lim)
        else:
            ax.set_xlim(-r_max * 1.1, r_max * 1.1)
            ax.set_ylim(-r_max * 1.1, r_max * 1.1)

        ax.set_aspect('equal')
        if filename is not None and mpl_ax is None:
            ax.set_ylabel('North (m)')
            ax.set_xlabel('East (m)')
            fig.savefig(filename)
            plt.close(fig)
        if filename is None and mpl_ax is None:
            plt.show()
            plt.close(fig)
        return ax

    @staticmethod
    def plot_taper(taper_func, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 8))
        r = np.linspace(0, 1, 100)
        y = taper_func(r, **kwargs)
        ax.plot(r, y)
        plt.show()
        plt.close(fig)

    def plot_min_sep(self, r_max, taper_func, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 8))
        r = np.linspace(0, 1, 100)
        y = self.station_diameter_m / taper_func(r, **kwargs)
        ax.plot(r * r_max, y)
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('Minimum separation (m)')
        plt.show()
        plt.close(fig)


class SKA1_low(Telescope):
    def __init__(self, name=''):
        Telescope.__init__(self, name)
        self.lon_deg = 116.63128900
        self.lat_deg = -26.69702400


def taylor_win(n, sll):
    """
    http://www.dsprelated.com/showcode/6.php

    from http://mathforum.org/kb/message.jspa?messageID=925929:

    A Taylor window is very similar to Chebychev weights. While Chebychev
    weights provide the tighest beamwidth for a given side-lobe level, taylor
    weights provide the least taper loss for a given sidelobe level.

    'Antenna Theory: Analysis and Design' by Constantine Balanis, 2nd
    edition, 1997, pp. 358-368, or 'Modern Antenna Design' by Thomas
    Milligan, 1985, pp.141-148.
    """
    def calculate_fm(m, sp2, a, nbar):
        n = np.arange(1, nbar)
        p = np.hstack([np.arange(1, m), np.arange(m + 1, nbar)])
        num = np.prod((1 - (m**2 / sp2) / (a**2 + (n - 0.5)**2)))
        den = np.prod(1 - m**2 / p**2)
        return ((-1)**(m + 1) * num) / (2 * den)
    nbar = int(np.ceil(2.0 * (acosh(10**(-sll / 20.0)) / pi)**2 + 0.5))
    n *= 2
    a = np.arccosh(10**(-sll / 20)) / pi
    sp2 = nbar**2 / (a**2 + (nbar - 0.5)**2)
    w = np.ones(n)
    fm = np.zeros(nbar)
    summation = 0
    k = np.arange(n)
    xi = (k - 0.5 * n + 0.5) / n
    for m in range(1, nbar):
        fm[m] = calculate_fm(m, sp2, a, nbar)
        summation += fm[m] * np.cos(2 * pi * m * xi)
    w += w * summation
    w /= w.max()
    w = w[n//2:]
    return w





