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
    out_dir = 'TEMP_results'
    name = 'ska1_v5'  # Name of the telescope model (prefix in file names)
    # -------------------------------------------------------------------------

    outer_radius = 6400

    if not isdir(out_dir):
        makedirs(out_dir)

    # Current SKA1 V5 design.
    tel = SKA1_low_analysis(name)
    tel.add_ska1_v5(r_max=outer_radius)

    # ----- Plots and metrics ---------
    # tel.save_iantconfig(join(out_dir, 'ska1_v5'))
    # tel.plot_layout(color='k', filename=join(out_dir, 'ska1_v5_layout.png'),
    #                 show_decorations=True)
    # tel.plot_grid(join(out_dir, '%s_uv_grid.png' % name),
    #               plot_radii=7e3)
    filename = join(out_dir, '%s_cables.png' % tel.name)
    l_ = tel.eval_cable_length(plot=True, plot_filename=filename, plot_r=7e3)
    np.savez(join(out_dir, '%s_metrics.npz' % tel.name),
             cable_length=l_)
    np.savetxt(join(out_dir, '%s_cables.txt' % tel.name),
               np.array([[0, l_]]),
               fmt=b'%.10f,%.10f')

if __name__ == '__main__':
    main()
