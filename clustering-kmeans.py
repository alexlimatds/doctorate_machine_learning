import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, davies_bouldin_score, adjusted_rand_score

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Removing instances with empty cols
#There is one instance with empty value in Money_Value col
df = df.dropna(subset=['Money_Value'])

#Getting ground truth
gtruth = df['Risk']

#Getting the selected cols
cols = ['PARA_A', 'PARA_B', 'Money_Value', 'numbers', 'Sector_score', 'District', 'History_score', 'LOSS_SCORE']
df = df[cols]

silhouettes = {}
def calcSilhouettes(preds, k):
  # Compute the silhouette scores for each instance
  sample_silhouette_values = silhouette_samples(df, preds)
  #iterate over clusters numbers
  for c_i in np.unique(preds):
    #getting silhouette of ith cluster
    s_mean = sample_silhouette_values[preds == c_i].mean()
    silhouettes[k][c_i].append(s_mean)

def printSilhouettes():
  log_file = open('kmeans-silhouettes.txt', 'w+')
  for k in silhouettes.keys():
    log_file.write('k={}\n'.format(k))
    log_file.write('cluster,silhouette_1,silhouette_2,silhouette_3,silhouette_4,silhouette_5\n')
    for c in range(len(silhouettes[k])):
      v = ','.join(map(str, silhouettes[k][c]))
      log_file.write('{},{}\n'.format(c, v))
  log_file.close()

dbs = {}
def calcDBs(preds, k):
  db = davies_bouldin_score(df, preds)
  dbs[k].append(db)

def printDBs():
  log_file = open('kmeans-DBs.txt', 'w+')
  log_file.write('k,DB_1,DB_2,DB_3,DB_4,DB_5\n')
  for k in dbs.keys():
    v = ','.join(map(str, dbs[k]))
    log_file.write('{},{}\n'.format(k, v))
  log_file.close()

crs = {}
def calcCRs(preds, k):
  cr = adjusted_rand_score(gtruth, preds)
  crs[k].append(cr)

def printCRs():
  log_file = open('kmeans-CRs.txt', 'w+')
  log_file.write('k,CR_1,CR_2,CR_3,CR_4,CR_5\n')
  for k in crs.keys():
    v = ','.join(map(str, crs[k]))
    log_file.write('{},{}\n'.format(k, v))
  log_file.close()
  
#The number of clusters will vary from 2 to 20. For each one cluster number, perform 5 experiments
for k in range(2, 21):
  silhouettes[k] = []
  dbs[k] = []
  crs[k] = []
  for i in range(0, k):
    silhouettes[k].insert(i, [])
  for i in range(1, 6):
    kmeans = KMeans(n_clusters=k, init='random', algorithm='full')
    predictions = kmeans.fit_predict(df)
    calcSilhouettes(predictions, k)
    calcDBs(predictions, k)
    calcCRs(predictions, k)

printSilhouettes()
printDBs()
printCRs()
