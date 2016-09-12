from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy
import matplotlib.pyplot as plt

# Load all text files into a nested dictionary.
data = {}
for model in range(3):
    f = b'cable_lengths/model%02i_cables.txt' % (model + 1)
    cables = numpy.loadtxt(f)
    for unwrap in range(9):
        f = b'iantconfig/Iantconfig_text_results/%02i%02i_snap.20x3.txt' % \
            (model + 1, unwrap)
        d = numpy.loadtxt(f, skiprows=1)
        t = {'baseline': d[:, 0], 'psfrms': d[:, 1], 'uvgap': d[:, 2],
             'unwrap_radius': cables[unwrap, 0],
             'cable_length': cables[unwrap, 1]}
        data['%02i%02i' % (model + 1, unwrap)] = t

# Plot PSFRMS.
_, ax = plt.subplots(figsize=(8, 8))
tel = '0100'
ax.plot(data[tel]['baseline'], data[tel]['psfrms'], 'b-', label=tel)
tel = '0108'
ax.plot(data[tel]['baseline'], data[tel]['psfrms'], 'r-', label=tel)
ax.legend()
ax.set_xlabel('Baseline length (m)')
ax.set_ylabel('PSFRMS')
plt.show()

# Plot UVGAP.
_, ax = plt.subplots(figsize=(8, 8))
tel = '0100'
ax.plot(data[tel]['baseline'], data[tel]['uvgap'], 'b-', label=tel)
tel = '0108'
ax.plot(data[tel]['baseline'], data[tel]['uvgap'], 'r-', label=tel)
ax.legend()
ax.set_xlabel('Baseline length (m)')
ax.set_ylabel('UVGAP')
plt.show()
