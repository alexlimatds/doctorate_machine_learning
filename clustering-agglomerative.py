import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
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
  clusters = np.unique(preds)
  avg = 0
  for c_i in clusters:
    #getting silhouette of ith cluster
    avg += sample_silhouette_values[preds == c_i].mean()
  avg = avg / clusters.size
  silhouettes[k].append(avg)

def printSilhouettes():
  log_file = open('agglomerative-silhouettes.txt', 'w+')
  log_file.write('k,silhouette_1\n')
  for k in silhouettes.keys():
    v = ','.join(map(str, silhouettes[k]))
    log_file.write('{},{}\n'.format(k, v))
  log_file.close()

dbs = {}
def calcDBs(preds, k):
  db = davies_bouldin_score(df, preds)
  dbs[k].append(db)

def printDBs():
  log_file = open('agglomerative-DBs.txt', 'w+')
  log_file.write('k,DB_1\n')
  for k in dbs.keys():
    log_file.write('{},{}\n'.format(k, dbs[k][0]))
  log_file.close()

crs = {}
def calcCRs(preds, k):
  cr = adjusted_rand_score(gtruth, preds)
  crs[k].append(cr)

def printCRs():
  log_file = open('agglomerative-CRs.txt', 'w+')
  log_file.write('k,CR_1\n')
  for k in crs.keys():
    log_file.write('{},{}\n'.format(k, crs[k][0]))
  log_file.close()
  
#The number of clusters will vary from 2 to 20
for k in range(2, 21):
  silhouettes[k] = []
  dbs[k] = []
  crs[k] = []
  algorithm = AgglomerativeClustering(n_clusters=k, linkage='average')
  predictions = algorithm.fit_predict(df)
  calcSilhouettes(predictions, k)
  calcDBs(predictions, k)
  calcCRs(predictions, k)

printSilhouettes()
printDBs()
printCRs()