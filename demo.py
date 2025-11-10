# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:20:01 2025

@author: Shuting Wang
"""

import sys
from pathlib import Path

# add src folder to sys.path (so Python can see it)
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from liquidbased_quantification import LiquidQuantifier

# now run
model = LiquidQuantifier("data/Monitoring_Data_Example.xlsx")
results = model.run_pipeline(show_plots=True)
