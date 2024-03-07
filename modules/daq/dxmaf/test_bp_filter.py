#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 5 2024

@author: grechc
"""
import sys
from DaqTimingPattern import *
import pydoocs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dl = pydoocs.read('XFEL.DIAG/TIMER.CENTRAL/MASTER/BUNCH_PATTERN')['data']

data = dl[::2]
data = data[:2708]
data = [decode_destination(unpack_timing_word(d)) for d in data]
data = {
            # see https://confluence.desy.de/display/DOOCS/DOOCS+Lectures+and+Tutorials?preview=%2F185871363%2F211286337%2FTiming+System+%26+Bunch+Pattern.pdf
            # contains all bunches with destination T4D that are not affected by a soft kick in TL
            'SA1' : np.where([dest_[0] == Destination['xfel'].T4D and dest_[1] != SpecialFlags['xfel'].TLD_SOFT_KICK for dest_ in data])[0],
            # contains all bunches with destination T5D.
            'SA2' : np.where([dest_[0] == Destination['xfel'].T5D for dest_ in data])[0],
            # contains all bunches with destination T4D that are affected by a soft kick in TL.
            'SA3' : np.where([dest_[0] == Destination['xfel'].T4D and dest_[1] == SpecialFlags['xfel'].TLD_SOFT_KICK for dest_ in data])[0]
}
print(data)