#This script generates artifacts used to analyse the selected data
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

df = pd.read_csv('trial.csv')

#Extracting selected columns
cols = ['PARA_A', 'PARA_B', 'Money_Value', 'numbers', 'Sector_score', 'District', 'History_score', 'LOSS_SCORE']
df = df[cols]

#Removing lines with empty value in Money_Value column
df = df.dropna(subset=['Money_Value'])

#getting selcte data
featureNames = list(df.columns.values)
x = df[featureNames]

#correlation matrix
plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=0.15)
#sns.heatmap(x.corr(), annot=True).figure.savefig('corr-selected-data.png')

#histograms
x.hist(bins=10, figsize=(9, 10), xlabelsize=10, ylabelsize=10, xrot=45, grid=False)
#pl.savefig('histograms-selected-data')

#plt.show()

#checking normality with Shapiro-Wilk Test
alpha = 0.05
print('*** Normality test - alpha = %.2f ***' % (alpha))
for col_name in list(cols):
	data = df[col_name]
	stat, p = shapiro(data)
	print('COLUMN: %s \n\tStatistics=%.3f, p=%.3f' % (col_name, stat, p))
	if p > alpha:
		print('\tSample looks Gaussian (fail to reject H0)')
	else:
		print('\tSample does not look Gaussian (reject H0)')