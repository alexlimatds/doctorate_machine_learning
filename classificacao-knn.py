import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Extracting columns to be used
cols = ['PARA_A', 'PARA_B', 'Money_Value', 'numbers', 'Sector_score', 'District', 'History_score', 'LOSS_SCORE', 'Risk']
df = df[cols]

#Removing instances with empty value to Money_Value column
df = df.dropna(subset=['Money_Value'])

#Extracting input vectors and class column
featureNames = list(df.columns.values)
featureNames.remove('Risk')
x = df[featureNames]
y = df.Risk

neighbors = [3, 9, 15, 30]

log_file = open('classification_scores-knn.txt', 'w+')

def report(scores, experimentName):
	print(experimentName)
	print('Mean accuracy on train: %0.2f' % (scores['train_score'].mean()))
	print('Standard deviation accuracy on train: %0.2f' % (scores['train_score'].std()))
	print('Mean accuracy on test: %0.2f' % (scores['test_score'].mean()))
	print('Standard deviation accuracy on test: %0.2f' % (scores['test_score'].std()))
	#writing test scores	
	log_file.write('{} score per fold\n'.format(experimentName))
	for s in scores['test_score']:
		log_file.write('{}\n'.format(s))

def expA(n):
	x_n = x
	experimento = '\n*** KNN - No Scaler - N={} - No weight***'.format(n)
	knn = KNeighborsClassifier(n_neighbors=n)
	#trainning
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#results
	report(cv_scores, experimento)

for i in neighbors:
	expA(i)

def expB(n):
	x_n = x
	experimento = '\n*** KNN - No Scaler - N={} - Weight: inverse of distance***'.format(n)
	knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
	#trainning
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#results
	report(cv_scores, experimento)

for i in neighbors:
	expB(i)

def expC(n):
	#normalização
	scaler = MinMaxScaler()
	scaler.fit(x)
	x_n = scaler.transform(x)
	experimento = '\n*** KNN - MinMax Scaler - N={} - No Weight***'.format(n)
	knn = KNeighborsClassifier(n_neighbors=n)
	#trainning
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#results
	report(cv_scores, experimento)

for i in neighbors:
	expC(i)

def expD(n):
	#normalização
	scaler = MinMaxScaler()
	scaler.fit(x)
	x_n = scaler.transform(x)
	experimento = '\n*** KNN - MinMax Scaler - N={} - Weight: inverse of distance***'.format(n)
	knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
	#trainning
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#results
	report(cv_scores, experimento)

for i in neighbors:
	expD(i)

log_file.close()
