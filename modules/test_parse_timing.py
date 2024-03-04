#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 19:09:12 2023

@author: weilun
"""
import sys

sys.path.append('/home/xfeloper/released_software/python/hlc_toolbox_common')
sys.path.append('/home/xfeloper/released_software/python/lib')
sys.path.append('/usr/share/libtine/python/python3.7m')
sys.path.append('/local/lib')

from DaqTimingPattern import unpack_timing_pattern
import pydoocs
import matplotlib.pyplot as plt
import pandas as pd


data = pydoocs.read('XFEL.DIAG/TIMER.CENTRAL/MASTER/BUNCH_PATTERN')['data']

x = unpack_timing_pattern(data)

dest_bit = data >> 18 & 15
softkick_bit = data >> 27 & 1
sa2_bit = data >> 26 & 1
t4_bit = data >> 25 & 1
dchp_bit = data >> 23 & 1


dest_dict = {'8': 'TLD',
        '4': 'T4D',
        '2': 'T5D',
        '0': 'Invalid'}

dest = [dest_dict[str(x)] for x in dest_bit[:4]]
softkick = softkick_bit[::4][201]
sa2 = sa2_bit[::4]
t4 = t4_bit[::4]
dchp = dchp_bit[::4]

pattern = pd.DataFrame({'dest': dest, "softkick": softkick, "t4": t4, "sa2": sa2, "dchp": dchp})