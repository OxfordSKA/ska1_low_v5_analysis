# -*- coding: utf-8 -*-
"""Module for evaluating interferometer sensitivity"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy.interpolate import interp1d
from math import sqrt, log, pi, radians, degrees
from astropy import constants as const
import matplotlib.pyplot as plt


def element_effective_area(freq_hz):
    """Return SKA1 Low element pattern effective area for given frequency

    Effective area values provided by Eloy de Lera Acedo
    (email: eloy .at. mrao.cam.ac.uk)

    Args:
        freq_hz (float): Frequency, in Hz

    Returns:
        Element effective area, in m^2
    """
    freqs = np.array([0.05e9, 0.07e9, 0.11e9, 0.17e9, 0.25e9, 0.35e9, 0.45e9,
                      0.55e9, 0.65e9])
    a_eff = np.array([1.8791, 1.8791, 1.8694, 1.3193, 0.6080, 0.2956, 0.2046,
                      0.1384, 0.0792])
    f_cut = 2
    f1 = interp1d(np.log10(freqs[:f_cut+1]), np.log10(a_eff[:f_cut+1]),
                  kind='slinear')
    f2 = interp1d(np.log10(freqs[f_cut:]), np.log10(a_eff[f_cut:]),
                  kind='cubic')
    if freq_hz <= freqs[f_cut]:
        return 10**f1(np.log10(freq_hz))
    else:
        return 10**f2(np.log10(freq_hz))


def system_temp(freq_hz):
    """Return SKA1 Low system temperatures for a given frequency.

    Values provided by Eloy de Lera Acedo
    (email: eloy .at. mrao.cam.ac.uk)

    Args:
        freq_hz (float): Frequency, in Hz

    Returns:
        System temperature, in K
    """
    freqs = np.array([0.05e9, 0.07e9, 0.11e9, 0.17e9, 0.25e9, 0.35e9, 0.45e9,
                      0.55e9, 0.65e9])
    t_sys = np.array([4.0409e3, 1.5029e3, 0.6676e3, 0.2936e3, 0.1402e3, 0.0873e3,
                      0.0689e3, 0.0607e3, 0.0613e3])
    f = interp1d(np.log10(freqs), np.log10(t_sys), kind='cubic')
    return 10**f(np.log10(freq_hz))


def flux_sensitivity(freq_hz, t_acc=5, bw_hz=100e3, num_antennas=256, eta=1,
                t_sys=None, a_eff=None):
    """Point source sensitivity a two element interferometer.
    For an unpolarised point source, single receiver polarisation.
    """
    t_sys = system_temp(freq_hz) if t_sys is None else t_sys
    a_eff = (element_effective_area(freq_hz) * num_antennas) if a_eff is None \
        else a_eff
    sefd = (2 * const.k_B.value * t_sys * eta) / a_eff
    sigma_s = (sefd * 1e26) / sqrt(2 * bw_hz * t_acc)
    return sigma_s


def brightness_temperature_sensitivity(freq_hz, beam_solid_angle, t_acc=5,
                                       bw_hz=100e3, num_antennas=256, eta=1,
                                       t_sys=None, a_eff=None):
    """Brightness temperature sensitivity"""
    sigma_s = flux_sensitivity(freq_hz, t_acc, bw_hz, num_antennas, eta, t_sys,
                               a_eff)
    factor = (1e-26 * const.c.value**2) / (2 * const.k_B.value * freq_hz**2)
    sigma_t = (sigma_s / beam_solid_angle) * factor
    return sigma_t


def beam_solid_angle(freq_hz, b_max, alpha=1.3):
    """Return beam solid angle, in sr and beam fwhm in arcsec.
    The value of alpha depends on the illumination (tapering) of the station.
    LOFAR, with no tapering has a value between 1.2 and 1.4 (ref:
    http://www.skatelescope.org/uploaded/59513_113_Memo_Nijboer.pdf, section 3)
    """
    wavelength = const.c.value / freq_hz
    fwhm = alpha * (wavelength / b_max)  # half power beam width
    sigma = fwhm / (2 * (2 * log(2))**2)
    solid_angle_sr = 2 * pi * sigma**2
    return solid_angle_sr, degrees(fwhm) * 3600

if __name__ == '__main__':
    freq_hz = 150e6
    sigma_s = flux_sensitivity(freq_hz=freq_hz, t_acc=1, bw_hz=1,
                               t_sys=60, a_eff=2.29e3)
    print(sigma_s, 'Jy')
    omega, fwhm = beam_solid_angle(freq_hz, 11e3)
    print(omega, fwhm)
    print('---------------')
    sigma_t = brightness_temperature_sensitivity(freq_hz, omega,
                                                 t_acc=1,
                                                 eta=1, bw_hz=1)
    print('---------------')
    print(sigma_t)
    # print((pi * radians(45 / 3600)**2) / (4 * log(2)))
