# -*- coding: utf-8 -*-
"""Module for a set of telescopes which we want to compare"""


class TelescopeSet(object):
    def __init__(self):
        self.telescopes = dict()

    def add_telescope(self, telescope):
        self.telescopes[telescope.name] = telescope

    def plot_my_metric(self):
        for name in self.telescopes:
            print(name)
