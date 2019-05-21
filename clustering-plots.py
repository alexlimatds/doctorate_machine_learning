import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker

k = list(range(2, 21))

#k-means
fig, (gDB, gCR, gSL) = plt.subplots(nrows=3)
fig.suptitle('k-means')

dbKmeans = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/kmeans-DBs.txt')
dbKmeans = dbKmeans[['DB_1', 'DB_2', 'DB_3', 'DB_4', 'DB_5']]
gDB.errorbar(k, dbKmeans.mean(axis=1), yerr=dbKmeans.std(axis=1), fmt='-o')
gDB.set(xlabel='k', ylabel='DB index')
gDB.margins(x=0.05)

crKmeans = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/kmeans-CRs.txt')
crKmeans = crKmeans[['CR_1', 'CR_2', 'CR_3', 'CR_4', 'CR_5']]
gCR.errorbar(k, crKmeans.mean(axis=1), yerr=crKmeans.std(axis=1), fmt='-o')
gCR.set(xlabel='k', ylabel='CR index')
gCR.margins(x=0.05)

slKmeans = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/kmeans-silhouettes.txt')
slKmeans = slKmeans[['silhouette_1', 'silhouette_2', 'silhouette_3', 'silhouette_4', 'silhouette_5']]
gSL.errorbar(k, slKmeans.mean(axis=1), yerr=slKmeans.std(axis=1), fmt='-o')
gSL.set(xlabel='k', ylabel='Silhouette index')
gSL.margins(x=0.05)

loc = plticker.MultipleLocator(base=1.0)
gDB.xaxis.set_major_locator(loc)
gCR.xaxis.set_major_locator(loc)
gSL.xaxis.set_major_locator(loc)
plt.show()