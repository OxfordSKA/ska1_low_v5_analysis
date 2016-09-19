from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy
import matplotlib.pyplot as plt
from os.path import join, isfile
from scipy.stats.mstats import gmean
import itertools


def main():
    marker = itertools.cycle(('v', 's', 'o'))
    num_models = 5

    # Load all text files into a nested dictionary.
    data = {}
    cable_length_dir = b'cable_lengths'

    # Load reference SKA layout.
    f = join(cable_length_dir, b'ska1_v5_cables.txt')
    cables = numpy.loadtxt(f)
    f = b'iantconfig/Iantconfig_text_results/ska1v5_snap.20x3.txt'
    d = numpy.loadtxt(f, skiprows=1)
    t = {'baseline': d[:, 0], 'psfrms': d[:, 1], 'uvgap': d[:, 2],
         'unwrap_radius': cables[0], 'cable_length': cables[1],
         'cable_length_delta': 0.0}
    data['ska1v5'] = t
    ref_cable_length = cables[1]
    ref_psfrms = gmean(d[:, 1])
    ref_uvgap = gmean(d[:, 2])

    # Load test models.
    for i_model in range(num_models):
        f = join(cable_length_dir, b'model%02i_cables.txt' % (i_model + 1))
        if isfile(f):
            cables = numpy.loadtxt(f)
        else:
            cables = numpy.zeros([9, 2])
        for i_unwrap in range(9):
            f = b'iantconfig/Iantconfig_text_results/%02i%02i_snap.20x3.txt' % \
                (i_model + 1, i_unwrap)
            d = numpy.loadtxt(f, skiprows=1)
            t = {'baseline': d[:, 0], 'psfrms': d[:, 1], 'uvgap': d[:, 2],
                 'unwrap_radius': cables[i_unwrap, 0],
                 'cable_length': cables[i_unwrap, 1],
                 'cable_length_delta': cables[i_unwrap, 1] - ref_cable_length,
                 'psfrms_ratio': gmean(d[:, 1]) / ref_psfrms,
                 'uvgap_ratio': gmean(d[:, 2]) / ref_uvgap}
            data['%02i%02i' % (i_model + 1, i_unwrap)] = t

    # Plot UVGAP for individual models.
    for i_model in range(num_models):
        _, ax = plt.subplots(figsize=(8, 8))
        for i_unwrap in range(9):
            t = data['%02i%02i' % (i_model + 1, i_unwrap)]
            ax.plot(t['baseline'], t['uvgap'], '-', marker=marker.next(),
                    linewidth=2, markersize=10,
                    label='Model %02i, unwrap %i' % (
                    i_model + 1, i_unwrap + 1))
        ax.legend(loc='best')
        ax.set_xlabel('Baseline length (m)')
        ax.set_ylabel('UVGAP')
        plt.show()

    # Plot UVGAP as a function of difference in cable cost
    # for each model series.
    keuro_cost_per_km = 1
    fig, ax = plt.subplots(figsize=(8, 8))
    for i_model in range(num_models):
        x = []
        y = []
        for i_unwrap in range(9):
            t = data['%02i%02i' % (i_model + 1, i_unwrap)]
            x.append((t['cable_length_delta'] / 1000) * keuro_cost_per_km)
            y.append(t['uvgap_ratio'])
        if numpy.sum(x) > 0:
            ax.plot(x, y, '-', marker=marker.next(), linewidth=2, markersize=10,
                    label='Model %i' % (i_model + 1))
    ax.legend()
    ax.set_xlabel('Additional cost, thousands of Euros (worst case)')
    ax.set_ylabel('UVGAP Ratio')
    ax.set_ylim((0, 1))
    plt.show()
    fig.savefig('uvgap_ratio_cost.png')


if __name__ == '__main__':
    main()
