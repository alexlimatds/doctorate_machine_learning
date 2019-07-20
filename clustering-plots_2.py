import pandas as pd
import pylab as pl
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker

k = list(range(2, 21))

#hierarchical
plt.xlabel('k')
plt.axhline(0, color='red')
plt.margins(x=0.05, y=0.05)
loc = plticker.MultipleLocator(base=1.0)
plt.xticks(k)
cr = pd.read_csv('agglomerative-CRs.txt')
db = pd.read_csv('agglomerative-DBs.txt')
sl = pd.read_csv('agglomerative-silhouettes.txt')
plt.title('Hierárquico')
plt.plot(k, cr['CR_1'], label='CR', marker='o')
plt.plot(k, db['DB_1'], label='DB', marker='o')
plt.plot(k, sl['silhouette_1'], label='Silhouette', marker='o')
pl.legend(loc='upper right')
pl.savefig('hierarchical_indexes_2')

#k-means
plt.clf()
plt.xlabel('k')
plt.margins(x=0.05, y=0.2)
plt.axhline(0, color='red')
loc = plticker.MultipleLocator(base=1.0)
plt.xticks(k)
cr = pd.read_csv('kmeans-CRs.txt')[['CR_1', 'CR_2', 'CR_3', 'CR_4', 'CR_5']]
db = pd.read_csv('kmeans-DBs.txt')[['DB_1', 'DB_2', 'DB_3', 'DB_4', 'DB_5']]
sl = pd.read_csv('kmeans-silhouettes.txt')[['silhouette_1', 'silhouette_2', 'silhouette_3', 'silhouette_4', 'silhouette_5']]
plt.title('k-means')
plt.errorbar(k, cr.mean(axis=1), yerr=cr.std(axis=1), label='CR', marker='o')
plt.errorbar(k, db.mean(axis=1), yerr=db.std(axis=1), label='DB', marker='o')
plt.errorbar(k, sl.mean(axis=1), yerr=sl.std(axis=1), label='Silhouette', marker='o')
pl.legend(ncol=3, loc='upper left')
pl.savefig('k-means_indexes_2')

#EM
cr = pd.read_csv('em-CRs.txt')[['CR_1', 'CR_2', 'CR_3', 'CR_4', 'CR_5']]
db = pd.read_csv('em-DBs.txt')[['DB_1', 'DB_2', 'DB_3', 'DB_4', 'DB_5']]
sl = pd.read_csv('em-silhouettes.txt')[['silhouette_1', 'silhouette_2', 'silhouette_3', 'silhouette_4', 'silhouette_5']]
#DB
plt.clf()
plt.xlabel('k')
plt.margins(x=0.05, y=0.05)
plt.axhline(0, color='red')
loc = plticker.MultipleLocator(base=1.0)
plt.xticks(k)
plt.title('EM')
plt.errorbar(k, db.mean(axis=1), yerr=db.std(axis=1), label='DB', marker='o')
pl.legend(ncol=3, loc='upper left')
pl.savefig('EM_indexes_2_DB')
#SL e CR
plt.clf()
plt.xlabel('k')
plt.margins(x=0.05, y=0.05)
plt.axhline(0, color='red')
loc = plticker.MultipleLocator(base=1.0)
plt.xticks(k)
plt.title('EM')
plt.errorbar(k, cr.mean(axis=1), yerr=cr.std(axis=1), label='CR', marker='o')
plt.errorbar(k, sl.mean(axis=1), yerr=sl.std(axis=1), label='Silhouette', marker='o')
pl.legend(ncol=3, loc='upper right')
pl.savefig('EM_indexes_2_SL_CR')