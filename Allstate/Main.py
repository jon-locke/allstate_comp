# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import csv
datafile = 'Data/test.csv'
data = pd.read_csv(datafile)
#Place algorithms Here
# example: SVMdata = ...


#Ensamble the data and set the output equal to the data to be outputted


output = data['loss']
with open('Results/result', 'w') as resultsfile:
    writer = csv.writer(resultsfile)
    writer.writerows(output)

