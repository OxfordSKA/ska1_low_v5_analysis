from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy
import matplotlib.pyplot as plt
from os.path import join, isfile
from scipy.stats.mstats import gmean
import itertools


def main():
    marker = itertools.cycle(('s', 'o'))
    colours = ('green', 'orange', 'black', 'blue', 'red')
    num_models = 5

    # Load all text files into a nested dictionary.
    data = {}
    cable_length_dir = b'cable_lengths'

    # Load reference SKA layout.
    f = join(cable_length_dir, b'ska1_v5_cables.txt')
    cables = numpy.loadtxt(f)
    for length in ('snap', '4hr'):
        f = b'iantconfig/Iantconfig_text_results/ska1v5_%s.20x3.txt' % length
        d = numpy.loadtxt(f, skiprows=1)
        t = {'baseline': d[:, 0],
             'psfrms': d[:, 1],
             'uvgap': d[:, 2],
             'unwrap_radius': cables[0],
             'cable_length': cables[1],
             'cable_length_delta': 0.0
             }
        data['ska1v5_%s' % length] = t

    # Load test models.
    for i_model in range(num_models):
        f = join(cable_length_dir, b'model%02i_cables.txt' % (i_model + 1))
        cables = numpy.loadtxt(f) if isfile(f) else numpy.zeros([9, 2])
        for length in ('snap', '4hr'):
            ref = data['ska1v5_%s' % length]
            for i_unwrap in range(9):
                f = b'iantconfig/Iantconfig_text_results/%02i%02i_%s.20x3.txt' \
                    % (i_model + 1, i_unwrap, length)
                d = numpy.loadtxt(f, skiprows=1)
                t = {'baseline': d[:, 0],
                     'psfrms': d[:, 1],
                     'uvgap': d[:, 2],
                     'unwrap_radius': cables[i_unwrap, 0],
                     'cable_length': cables[i_unwrap, 1],
                     'cable_length_delta':
                         cables[i_unwrap, 1] - ref['cable_length'],
                     'psfrms_ratio': gmean(d[:, 1]) / gmean(ref['psfrms']),
                     'uvgap_ratio': gmean(d[:, 2]) / gmean(ref['uvgap'])
                     }
                data['%02i%02i_%s' % (i_model + 1, i_unwrap, length)] = t

    # Plot UVGAP for individual models.
    # for i_model in range(num_models):
    #     _, ax = plt.subplots(figsize=(8, 8))
    #     for length in ('snap', '4hr'):
    #         for i_unwrap in range(9):
    #             t = data['%02i%02i_%s' % (i_model + 1, i_unwrap, length)]
    #             ax.plot(t['baseline'], t['uvgap'], '-', marker=marker.next(),
    #                     linewidth=2, markersize=8,
    #                     label='Model %02i, unwrap %i, length %s' % (
    #                     i_model + 1, i_unwrap + 1, length))
    #         ax.legend(loc='best')
    #         ax.set_xlabel('Baseline length (m)')
    #         ax.set_ylabel('UVGAP')
    #         plt.show()

    # Plot UVGAP as a function of difference in cable cost
    # for each model series.
    keuro_cost_per_km = 1
    fig, ax = plt.subplots(figsize=(8, 8))
    for i_model in range(num_models):
        for length in ('snap', '4hr'):
            x = []
            y = []
            for i_unwrap in range(9):
                t = data['%02i%02i_%s' % (i_model + 1, i_unwrap, length)]
                x.append((t['cable_length_delta'] / 1000) * keuro_cost_per_km)
                y.append(t['uvgap_ratio'])
            label = 'Model %i' % (i_model + 1)
            if length == 'snap':
                label += ', snapshot'
            elif length == '4hr':
                label += ', 4-hour'
            linestyle = '--' if length == 'snap' else '-'
            if numpy.sum(x) > 0:
                ax.plot(x, y, linestyle, marker=marker.next(), linewidth=2,
                        color=colours[i_model],
                        markersize=8, label=label)
    ax.legend(ncol=2, loc='best')
    ax.set_xlabel('Additional cost, thousands of Euros (worst case)')
    ax.set_ylabel('UVGAP Ratio')
    ax.grid(b=True, which='major', color='darkgrey', linestyle='-')
    ax.grid(b=True, which='minor', color='lightgrey', linestyle='-')
    ax.set_ylim((0, 1))
    plt.show()
    fig.savefig('uvgap_ratio_cost.png')


if __name__ == '__main__':
    main()
