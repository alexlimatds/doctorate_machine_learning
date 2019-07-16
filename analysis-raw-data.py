#This script generates artifacts used to analyse the raw data
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Removing categorical/nominal columns
df = df.drop(['LOCATION_ID', 'Risk'], axis=1)

#Removing lines with empty value in Money_Value column
df = df.dropna(subset=['Money_Value'])

#correlation matrix
featureNames = list(df.columns.values)
x = df[featureNames]

plt.figure(figsize=(14, 15))
plt.subplots_adjust(bottom=0.15)
sns.heatmap(x.corr(), annot=True).figure.savefig('corr-raw-data.png')
plt.show()