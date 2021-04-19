#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:15:15 2020

@author: molano
"""
from ngym_shaping.wrappers.shaping import Shaping


ALL_WRAPPERS = {'shaping-v0':
                'ngym_shaping.wrappers.shaping:Shaping'}


def all_wrappers():
    return sorted(list(ALL_WRAPPERS.keys()))
