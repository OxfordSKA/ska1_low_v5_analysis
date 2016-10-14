#!/usr/bin/python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import numpy
import matplotlib.pyplot as plt


def main():
    core_radius = 450.0
    overlap = 0.3
    current_radius = core_radius * 0.5
    max_radius = 1700.0

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Add the auto-convolved core at the origin.
    # ax.add_artist(plt.Circle((0, 0), 1.5*core_radius, fill=False, color='b'))

    while current_radius < (max_radius - core_radius):
        current_radius += 2 * core_radius * (1.0 - overlap)
        circumference = 2 * math.pi * current_radius
        num_in_ring = int(math.ceil(circumference /
                                    (2 * core_radius * (1.0 - overlap))))
        if num_in_ring % 2:
            num_in_ring -= 1
        print('Radius: %.2f, Number: %d' % (current_radius, num_in_ring))
        delta_theta = 2 * math.pi / num_in_ring
        for i in range(num_in_ring):
            x = current_radius * math.cos(i * delta_theta)
            y = current_radius * math.sin(i * delta_theta)
            ax.add_artist(
                plt.Circle((x, y), core_radius,
                           fill=True, color='r', alpha=0.2, lw=0))

    plot_limit = max_radius * 1.1 + 2 * core_radius
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    plt.show()

def main2():
    core_radius = 450.0
    max_radius = 1700.0

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Add the auto-convolved core at the origin.
    # ax.add_artist(plt.Circle((0, 0), 1.5*core_radius, fill=False, color='b'))

    # Randall's points
    radii = [800, 1250, 1700]
    num_in_ring = numpy.array([12, 24, 36])

    # Odd
    radii = [900, 1700]
    radii = [1000, 1700]
    num_in_ring = 2 * numpy.array([29, 43])

    # Even
    # radii = [900, 1700]
    # num_in_ring = numpy.array([28, 44])
    for i_radius in range(len(radii)):
        current_radius = radii[i_radius]
        delta_theta = 2 * math.pi / num_in_ring[i_radius]
        for i in range(num_in_ring[i_radius]):
            x = current_radius * math.cos(i * delta_theta)
            y = current_radius * math.sin(i * delta_theta)
            ax.add_artist(
                plt.Circle((x, y), core_radius,
                           fill=True, color='r', alpha=0.1, lw=0))

    plot_limit = max_radius * 1.1 + 2 * core_radius
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    plt.show()


if __name__ == '__main__':
    main2()