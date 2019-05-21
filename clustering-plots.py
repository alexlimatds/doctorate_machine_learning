import pandas as pd
import pylab as pl
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker

k = list(range(2, 21))

def subplotError(indexName, y_values, errors, sub):
  sub.errorbar(k, y_values, errors, fmt='-o')
  sub.set(xlabel='k', ylabel=indexName)
  sub.margins(x=0.05)

def subplot(indexName, y_values, sub):
  sub.plot(k, y_values, marker='o')
  sub.set(xlabel='k', ylabel=indexName)
  sub.margins(x=0.05)

def plot(methodName, dbPath, crPath, slPath, withError=False):
  fig, (gDB, gCR, gSL) = plt.subplots(nrows=3)
  fig.suptitle(methodName)

  dbKmeans = pd.read_csv(dbPath)
  crKmeans = pd.read_csv(crPath)
  slKmeans = pd.read_csv(slPath)
  
  if withError:
    dbKmeans = dbKmeans[['DB_1', 'DB_2', 'DB_3', 'DB_4', 'DB_5']]
    subplotError('DB index', dbKmeans.mean(axis=1), dbKmeans.std(axis=1), gDB)
    
    crKmeans = crKmeans[['CR_1', 'CR_2', 'CR_3', 'CR_4', 'CR_5']]
    subplotError('CR index', crKmeans.mean(axis=1), crKmeans.std(axis=1), gCR)
    
    slKmeans = slKmeans[['silhouette_1', 'silhouette_2', 'silhouette_3', 'silhouette_4', 'silhouette_5']]
    subplotError('Silhouette index', slKmeans.mean(axis=1), slKmeans.std(axis=1), gSL)
  else:
    subplot('DB index', dbKmeans['DB_1'], gDB)
    subplot('CR index', crKmeans['CR_1'], gCR)
    subplot('Silhouette index', slKmeans['silhouette_1'], gSL)

  loc = plticker.MultipleLocator(base=1.0)
  gDB.xaxis.set_major_locator(loc)
  gCR.xaxis.set_major_locator(loc)
  gSL.xaxis.set_major_locator(loc)
  pl.savefig('{}_indexes'.format(methodName))

plot('k-means', 'kmeans-DBs.txt', 'kmeans-CRs.txt', 'kmeans-silhouettes.txt', withError=True)
plot('EM', 'em-DBs.txt', 'em-CRs.txt', 'em-silhouettes.txt', withError=True)
plot('hierarchical', 'agglomerative-DBs.txt', 'agglomerative-CRs.txt', 'agglomerative-silhouettes.txt')