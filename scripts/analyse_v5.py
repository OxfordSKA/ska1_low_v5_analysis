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
    name = 'ska1_v5'  # Name of the telescope model (prefix in file names)
    # -------------------------------------------------------------------------

    if not isdir(out_dir):
        makedirs(out_dir)

    # Current SKA1 V5 design.
    tel = SKA1_low_analysis()
    r_cut = 6400
    tel.add_ska1_v5(None, r_cut)
    # tel.save_iantconfig(join(out_dir, 'ska1_v5'))
    # tel.plot_layout(color='k', filename=join(out_dir, 'ska1_v5_layout.png'),
    #                 show_decorations=True)
    tel.plot_grid(join(out_dir, '%s_uv_grid.png' % name),
                  x_lim=[-r_cut*2, r_cut*2],
                  y_lim=[-r_cut*2, r_cut*2], plot_radii=[r_cut*2])

    # ref_cable_length = tel.cable_length(
    #     plot=True, plot_filename=join(out_dir, 'cable_length_v5.png'))

if __name__ == '__main__':
    main()
