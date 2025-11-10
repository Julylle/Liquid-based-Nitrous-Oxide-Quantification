# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:20:01 2025

@author: Shuting Wang
"""
#%%
from liquidbased_quantification import LiquidQuantifier

model = LiquidQuantifier("data/Monitoring_Data_Example.xlsx")
results = model.run_pipeline(show_plots=True)
print(results.head())
results.to_excel("liquid_quantification_results.xlsx", index=False)

