# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import sys
import time
from math import sqrt, ceil, radians, cos, sin

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np


class Layout(object):
    """Class for generating 2d distributions of points"""

    def __init__(self, seed=None, trail_timeout=2.0, num_trials=5):
        self.x = None
        self.y = None
        self.info = None
        self.seed = seed
        self.trial_timeout = trail_timeout
        self.num_trials = num_trials
        self.poly_mask_path = list()

    def clear(self):
        self.x = None
        self.y = None
        self.info = None
        self.poly_mask_path = list()

    @staticmethod
    def rotate_coords(x, y, angle):
        """ Rotate coordinates counter clockwise by angle, in degrees.
        Args:
            x (array like): array of x coordinates.
            y (array like): array of y coordinates.
            angle (float): Rotation angle, in degrees.

        Returns:
            (x, y) tuple of rotated coordinates

        """
        theta = radians(angle)
        xr = x * np.cos(theta) - y * np.sin(theta)
        yr = x * np.sin(theta) + y * np.cos(theta)
        return xr, yr

    @staticmethod
    def polygon_vertices_(num_sides, radius, theta_0_deg):
        """Return polygon vertices on a circumscribed circle of given radius"""
        angles = np.arange(0, 360, 360 / num_sides) + theta_0_deg
        x = radius * np.cos(np.radians(angles))
        y = radius * np.sin(np.radians(angles))
        x = np.append(x, 0.0)
        y = np.append(y, 0.0)
        codes = [Path.MOVETO]
        for i in range(num_sides - 1):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        return np.array(zip(x, y)), codes

    def apply_poly_mask(self, num_sides, radius, theta_0_deg=0, invert=False,
                        pad_radius=0):
        """Remove data points outside polygonal mask

        The pad radius can be used to exclude points this distance from the
        edge of the polygon
        """
        verts, codes = Layout.polygon_vertices_(num_sides, radius - pad_radius,
                                                theta_0_deg)
        points = np.vstack((self.x, self.y)).T
        path = Path(verts, codes=codes)
        mask_idx = path.contains_points(points)
        if invert:
            mask_idx = np.invert(mask_idx)
        self.x = self.x[mask_idx]
        self.y = self.y[mask_idx]
        if pad_radius != 0:
            verts, codes = Layout.polygon_vertices_(num_sides, radius,
                                                    theta_0_deg)
            path = Path(verts, codes=codes)
        self.poly_mask_path.append(path)




    @staticmethod
    def log_spiral_1(r0, b, delta_theta_deg, n):
        """Computes coordinates on a log spiral.

        Args:
            r0 (float): minimum radius
            b (float): Spiral constant.
            delta_theta_deg (float): angle between points, in degrees.
            n (int): Number of points.

        Returns:
            tuple: (x, y) coordinates
        """
        t = np.arange(n) * math.radians(delta_theta_deg)
        tmp = r0 * np.exp(b * t)
        x = tmp * np.cos(t)
        y = tmp * np.sin(t)
        return x, y

    @staticmethod
    def log_spiral_2(r0, r1, b, n):
        """Computes coordinates on a log spiral.

        Args:
            r0 (float): minimum radius
            r1 (float): maximum radius
            b (float): Spiral constant.
            n (int): Number of points.

        Returns:
            tuple: (x, y) coordinates
        """
        if b == 0.0:
            x = np.exp(np.linspace(math.log(r0), math.log(r1), n))
            y = np.zeros(n)
        else:
            t_max = math.log(r1 / r0) * (1.0 / b)
            t = np.linspace(0, t_max, n)
            tmp = r0 * np.exp(b * t)
            x = tmp * np.cos(t)
            y = tmp * np.sin(t)
        return x, y

    @staticmethod
    def log_spiral_clusters(r0, r1, b, n, n_cluster, r_cluster, min_sep,
                            cluster_timeout=2.0, tries_per_cluster=5,
                            seed=None):
        """Computes coordinates on a log spiral.

        Args:
            r0 (float): minimum radius
            r1 (float): maximum radius
            b (float): Spiral constant.
            n (int): Number of points.
            n_cluster (int): Number of points per cluster
            r_cluster (double): Radius of the cluster.
            min_sep (double): minimum separation of points in each cluster.
            cluster_timeout (Optional[double]): timeout per cluster, in seconds
            tries_per_cluster (Optional[int]): number of seeds to try
            seed (Optional[int]): Random number seed.

        Returns:
            tuple: (x, y) coordinates
        """
        seed = np.random.randint(1, 1e5) if not seed else seed
        if b == 0.0:
            x = np.exp(np.linspace(math.log(r0), math.log(r1), n))
            y = np.zeros(n)
        else:
            t_max = math.log(r1 / r0) * (1.0 / b)
            t = np.linspace(0, t_max, n)
            tmp = r0 * np.exp(b * t)
            x = tmp * np.cos(t)
            y = tmp * np.sin(t)
        x_all = np.zeros((n, n_cluster))
        y_all = np.zeros((n, n_cluster))
        max_iter, max_time_taken, max_tries, max_total_tries = 0, 0.0, 0, 0
        for k in range(n):
            for t in range(tries_per_cluster):
                sys.stdout.flush()
                np.random.seed(seed + t)
                xc, yc, tries, time_taken, total_tries = \
                    Layout.rand_uniform_2d(n_cluster, r_cluster, min_sep,
                                    timeout=cluster_timeout)
                max_iter = max(max_iter, tries)
                max_time_taken = max(max_time_taken, time_taken)
                max_total_tries = max(max_total_tries, total_tries)
                if xc.shape[0] == n_cluster:
                    max_tries = max(max_tries, t)
                    break
            if not xc.shape[0] == n_cluster:
                raise RuntimeError('Failed to generate cluster [%i]. '
                                   '%i/%i stations generated. '
                                   '(max tries for 1 point = %i, '
                                   'max total tries = %i)'
                                   % (k, xc.shape[0], n_cluster, max_iter,
                                      max_total_tries))
            x_all[k, :], y_all[k, :] = xc + x[k], yc + y[k]
        sys.stdout.flush()
        return x_all.flatten(), y_all.flatten(), x, y

    @staticmethod
    def grid_position(x, y, scale, r):
        ix = int(round(x + r) * scale)
        iy = int(round(y + r) * scale)
        return ix, iy

    @staticmethod
    def get_trial_position(r):
        return tuple(np.random.rand(2) * 2.0 * r - r)

    @staticmethod
    def rand_uniform_2d(n, r_max, min_sep, timeout, r_min=0.0):
        """
        Generate 2d random points with a minimum separation within a radius
        range specified by r_max and r_min.

        Args:
            n (int): Number of points to generate.
            r_max (float):  Maximum radius.
            min_sep (float): Minimum separation of points.
            timeout (Optional[float]): timeout, in seconds.
            r_min (Optional[float]): Minimum radius.
            seed (Optional[int]): random number seed

        Returns:
            tuple (x, y, info)

        """
        grid_size = min(100, int(ceil(float(r_max * 2.0) / min_sep)))
        grid_cell = float(r_max * 2.0) / grid_size  # Grid sector size
        scale = 1.0 / grid_cell  # Scaling onto the sector grid.

        x, y = np.zeros(n), np.zeros(n)
        grid = {
            'start': np.zeros((grid_size, grid_size), dtype='i8'),
            'end': np.zeros((grid_size, grid_size), dtype='i8'),
            'count': np.zeros((grid_size, grid_size), dtype='i8'),
            'next': np.zeros(n, dtype='i8')
        }

        t0 = time.time()
        n_generated = n
        num_tries, max_tries, total_tries = 0, 0, 0
        for j in range(n):
            done = False
            while not done:
                xt, yt = Layout.get_trial_position(r_max)
                rt = (xt**2 + yt**2)**0.5
                # Point is inside area defined by: r_min < r < r_max
                if rt + (min_sep / 2.0) > r_max:
                    num_tries += 1
                elif r_min and rt - (min_sep / 2.0) < r_min:
                    num_tries += 1
                else:
                    jx, jy = Layout.grid_position(xt, yt, scale, r_max)
                    y0 = max(0, jy - 2)
                    y1 = min(grid_size, jy + 3)
                    x0 = max(0, jx - 2)
                    x1 = min(grid_size, jx + 3)

                    # Find minimum spacing between trial and other points.
                    d_min = r_max * 2.0
                    for ky in range(y0, y1):
                        for kx in range(x0, x1):
                            if grid['count'][ky, kx] > 0:
                                i_other = grid['start'][ky, kx]
                                for kh in range(grid['count'][ky, kx]):
                                    dx = xt - x[i_other]
                                    dy = yt - y[i_other]
                                    d_other = (dx**2 + dy**2)**0.5
                                    d_min = min(d_min, d_other)
                                    i_other = grid['next'][i_other]

                    if d_min >= min_sep:
                        x[j], y[j] = xt, yt
                        if grid['count'][jy, jx] == 0:
                            grid['start'][jy, jx] = j
                        else:
                            grid['next'][grid['end'][jy, jx]] = j
                        grid['end'][jy, jx] = j
                        grid['count'][jy, jx] += 1
                        max_tries = max(max_tries, num_tries)
                        total_tries += num_tries
                        num_tries = 0
                        done = True
                    else:
                        num_tries += 1

                if (time.time() - t0) >= timeout:
                    max_tries = max(max_tries, num_tries)
                    total_tries += num_tries
                    n_generated = j - 1
                    done = True

            if (time.time() - t0) >= timeout:
                max_tries = max(max_tries, num_tries)
                total_tries += num_tries
                break

        if n_generated < n:
            x, y = x[:n_generated], y[:n_generated]

        return x, y, {'max_tries': max_tries,
                      'total_tries': total_tries,
                      'time_taken': time.time() - t0}

    @staticmethod
    def rand_uniform_2d_trials(n, r_max, min_sep, trial_timeout, num_trials,
                               seed0, r_min=0.0):
        max_generated = 0
        all_info = dict()
        t0 = time.time()
        for t in range(num_trials):
            np.random.seed(seed0 + t)
            x, y, info = Layout.rand_uniform_2d(n, r_max, min_sep,
                                                trial_timeout, r_min)
            all_info[t] = info
            all_info[t]['seed'] = seed0 + t
            all_info[t]['num_generated'] = x.shape[0]
            if x.shape[0] == n:
                all_info['attempt_id'] = t
                all_info['total_time'] = time.time() - t0
                all_info['final_seed'] = seed0 + t
                return x, y, all_info
            else:
                max_generated = max(max_generated, x.shape[0])
        raise RuntimeError('Failed to generate enough points. '
                           'max generated: %i / %i' % (max_generated, n))

    @staticmethod
    def rand_uniform_2d_trials_r_max(n, r_max, min_sep, trial_timeout,
                                     num_trials=5, seed=None,
                                     r_min=0.0, verbose=False):
        if r_max.shape[0] == 0:
            raise AssertionError('rmax must be an array of r max values')

        seed = np.random.randint(1, 1e8) if not seed else seed
        max_generated = 0
        all_info = dict()
        t0 = time.time()
        for ir, r in enumerate(r_max):
            all_info[r] = dict()
            if verbose:
                print('(%-3i/%3i) %8.3f' % (ir, r_max.shape[0], r), end=': ')
            for t in range(num_trials):
                np.random.seed(seed + t)
                x, y, info = Layout.rand_uniform_2d(n, r, min_sep, trial_timeout, r_min)
                all_info[r][t] = info
                all_info[r][t]['seed'] = seed + t
                all_info[r][t]['num_generated'] = x.shape[0]
                if x.shape[0] == n:
                    all_info['attempt_id'] = t
                    all_info['total_time'] = time.time() - t0
                    all_info['final_seed'] = seed + t
                    all_info['final_radius'] = r
                    all_info['final_radius_id'] = ir
                    all_info['r_max'] = r_max
                    if verbose:
                        print('%i' % x.shape[0])
                    return x, y, all_info
                else:
                    max_generated = max(max_generated, x.shape[0])
                    if verbose:
                        print('%i%s' % (x.shape[0], ',' if t < num_trials-1 else ''),
                              end='')
            if verbose:
                print(' ')
        if verbose:
            print('%i' % x.shape[0])
            sys.stdout.flush()
            for key in all_info:
                print(key, ':', all_info[key])
                sys.stdout.flush()

        raise RuntimeError('Failed to generate enough points. '
                           'max generated: %i / %i' % (max_generated, n))

    @staticmethod
    def rand_uniform_2d_tapered_(n, r_max, min_sep, taper_func, timeout, r_min,
                                 **kwargs):
        """
        Generate 2d random points with a minimum separation within a radius
        range specified by r_max and r_min.

        Args:
            n (int): Number of points to generate.
            r_max (float):  Maximum radius.
            min_sep (float): Minimum separation of points before modification by
                             taper function.
            taper_func (function): Taper function for min sep growth.
            timeout: timeout, in seconds.
            r_min: Minimum radius.
            seed: random number seed
            **kwargs: arguments passed to the taper function

        Returns:
            tuple (x, y, info)

        """
        # Range of separation of points at the inner and outer generation radius.
        # FIXME(BM) these are an approximation.
        r_min_ = (r_min + (min_sep / 2)) / r_max
        r_max_ = (r_max - (min_sep / 2)) / r_max
        min_sep_r_min = min_sep / taper_func(r_max_, **kwargs)
        min_sep_r_max = min_sep / taper_func(r_max_, **kwargs)

        grid_size = min(100, int(ceil(float(r_max * 2.0) / min_sep_r_max)))
        grid_cell = float(r_max * 2.0) / grid_size  # Grid sector size
        scale = 1.0 / grid_cell  # Scaling onto the sector grid.

        x, y, i_min = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int)
        min_dist = np.zeros(n)
        grid = {
            'start': np.zeros((grid_size, grid_size), dtype=np.int),
            'end': np.zeros((grid_size, grid_size), dtype=np.int),
            'count': np.zeros((grid_size, grid_size), dtype=np.int),
            'next': np.zeros(n, dtype=np.int)
        }

        # FIXME(BM) only store the trials for the last point...
        trials = np.ones((int(1e7), 2))
        t0 = time.time()
        n_generated = n
        num_tries, num_tries_last, max_tries, total_tries = 0, 0, 0, 0
        for j in range(n):
            done = False
            while not done:
                xt, yt = Layout.get_trial_position(r_max)
                total_tries += 1
                num_tries += 1
                r_trial = (xt**2 + yt**2)**0.5
                trials[num_tries, :] = xt, yt
                # FIXME(BM) use 'min_dist_trial' here... ?
                # Check if trial point meets the condition: r_min < r < r_max
                if r_trial + (min_sep_r_max / 2.0) >= r_max:
                    continue
                elif r_min and r_trial - (min_sep_r_min / 2.0) < r_min:
                    continue
                else:
                    t_ = 1 / taper_func(r_trial / r_max, **kwargs)
                    min_dist_trial = min_sep * t_

                    # Grid cell the trail falls into
                    gx, gy = Layout.grid_position(xt, yt, scale, r_max)

                    # Range of nearby grid cells we need to check.
                    border = 1  # FIXME(BM) check this value is ok
                    y0 = max(0, gy - border)
                    y1 = min(grid_size, gy + border + 1)
                    x0 = max(0, gx - border)
                    x1 = min(grid_size, gx + border + 1)

                    # Loop over nearby grid cells to check if any nearby point
                    # overlaps with this trial position given scaled minimum
                    # separations.
                    i_min_trial = -1
                    min_separation = r_max * 2.0
                    overlap_found = False
                    for ky in range(y0, y1):
                        for kx in range(x0, x1):
                            if grid['count'][ky, kx] > 0:
                                i_other = grid['start'][ky, kx]
                                for kh in range(grid['count'][ky, kx]):
                                    dx = xt - x[i_other]
                                    dy = yt - y[i_other]
                                    separation = (dx**2 + dy**2)**0.5
                                    separation -= min_dist[i_other] / 2
                                    separation -= min_dist_trial / 2
                                    if separation < min_separation:
                                        min_separation = separation
                                        i_min_trial = i_other
                                    i_other = grid['next'][i_other]

                    if min_separation >= 0:
                        x[j], y[j], i_min[j] = xt, yt, i_min_trial
                        min_dist[j] = min_dist_trial
                        if grid['count'][gy, gx] == 0:
                            grid['start'][gy, gx] = j
                        else:
                            grid['next'][grid['end'][gy, gx]] = j
                        grid['end'][gy, gx] = j
                        grid['count'][gy, gx] += 1
                        max_tries = max(max_tries, num_tries)
                        num_tries_last = num_tries
                        num_tries = 0
                        done = True

                if (time.time() - t0) >= timeout:
                    max_tries = max(max_tries, num_tries)
                    n_generated = j - 1
                    done = True

            if (time.time() - t0) >= timeout:
                max_tries = max(max_tries, num_tries)
                break

        if n_generated < n:
            # print(total_tries, max_tries, n_generated, n)
            x, y = x[:n_generated], y[:n_generated]

        # FIXME(BM) rename to tries for last point (used to plot if failed to see how hard the last point was pushed...)
        trials = trials[:num_tries_last]
        return x, y, {
            'i_min': i_min,
            'max_tries': max_tries,
            'total_tries': total_tries,
            'time_taken': time.time() - t0,
            'trials': trials}

    @classmethod
    def rand_tapered_2d_trials(cls, n, r_max, r_min, min_sep, trial_timeout,
                               num_trials, seed0, taper_func, **kwargs):
        """
        Generate a random 2d layout of n points, returning an instance of the
        Layout class.

        The points are generated with a random uniform distribution constrained
        by a minimum separation which grows as a function of radius according
        to the specified taper function.

        Args:
            n (int): number of points to generate.
            r_max (float): Maximum radius of points
            r_min (float): Minimum radius of points.
            min_sep (float): Minimum separation of points at radius == 0
            trial_timeout (float): Time, in seconds, to spend on each trial.
            num_trials (int): Number of trials of incrementing seed.
            seed0 (int): Initial random number seed (for trail 0)
            taper_func (function): Taper function.
            **kwargs: Variable argument list to be passed to the taper function.

        Returns:
            Layout class.
        """
        layout = cls()
        max_generated = 0
        all_info = dict()
        t0 = time.time()
        for t in range(num_trials):
            print('.', end='')
            np.random.seed(seed0 + t)
            x, y, info = Layout.rand_uniform_2d_tapered_(
                n, r_max, min_sep, taper_func, trial_timeout, r_min, **kwargs)
            all_info[t] = info
            all_info[t]['seed'] = seed0 + t
            all_info[t]['num_generated'] = x.shape[0]
            if x.shape[0] == n:
                all_info['attempt_id'] = t
                all_info['total_time'] = time.time() - t0
                all_info['final_seed'] = seed0 + t
                layout.x = x
                layout.y = y
                layout.info = all_info
                return layout
            else:
                max_generated = max(max_generated, x.shape[0])
        print('')
        layout.info = all_info
        raise RuntimeError('Failed to generate enough points. '
                           'max generated: %i / %i.' % (max_generated, n))

    @staticmethod
    def generate_clusters(num_clusters, n, r_max, min_sep, trail_timeout,
                          num_trials, seed0, r_min = 0.0):
        x_all, y_all, info_all = list(), list(), list()
        for i in range(num_clusters):
            seed = seed0 + i * num_trials
            x, y, info = Layout.rand_uniform_2d_trials(
                n, r_max, min_sep, trail_timeout, num_trials, seed, r_min)
            if x.shape[0] != n:
                raise RuntimeError('Failed to generate cluster %i' % i)
            x_all.append(x)
            y_all.append(y)
            info_all.append(info)
        return x_all, y_all, info_all

    def uniform_cluster(self, num_points, min_sep, r_max, r_min):
        """Generate a uniform random cluster (core) of points"""
        if self.seed is None:
            self.seed = np.random.randint(1, 1e8)
        x, y, _ = self.rand_uniform_2d_trials(
            num_points, r_max, min_sep, self.trial_timeout,
            self.num_trials, self.seed, r_min)
        if not x.size == num_points:
            raise RuntimeError('Failed to generate enough points')
        self.x = x
        self.y = y

    def hex_lattice(self, separation, r_max, theta0_deg=0.0):
        n = int(ceil(r_max * 3 / separation))
        w = separation
        h = w * sqrt(3) / 2
        x = np.zeros((n, n))
        y = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                x[j, i] = (i + 0.5 * j) * w
                y[j, i] = j * h
        x -= x[n // 2, n // 2]
        y -= y[n // 2, n // 2]
        x = x.flatten()
        y = y.flatten()
        x, y = Layout.rotate_coords(x, y, theta0_deg)
        r = (x**2 + y**2)**0.5
        idx = np.where(r <= (r_max - separation / 2))[0]
        self.x = x[idx]
        self.y = y[idx]

    def plot(self, plot_radius=None, plot_radii=[]):
        if self.x is None or self.y is None:
            return
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.plot(self.x, self.y, 'k+')

        for r in plot_radii:
            color = r[1] if isinstance(r, tuple) else 'r'
            radius = r[0] if isinstance(r, tuple) else r
            ax.add_artist(plt.Circle((0, 0), radius, fill=False, color=color))

        for path in self.poly_mask_path:
            patch = patches.PathPatch(path, facecolor='None', lw=1)
            ax.add_patch(patch)

        if plot_radius:
            ax.set_xlim(-plot_radius, plot_radius)
            ax.set_ylim(-plot_radius, plot_radius)

        plt.show()
        plt.close(fig)
