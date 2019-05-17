import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Extração de colunas a serem utilizadas
cols = ['PARA_A', 'PARA_B', 'Money_Value', 'numbers', 'Sector_score', 'District', 'History_score', 'LOSS_SCORE', 'Risk']
df = df[cols]

#Remoção de instâncias com colunas vazias
#Há uma instância com valor vazion para a coluna Money_Value
df = df.dropna(subset=['Money_Value'])

#extração dos vetores e das classes
featureNames = list(df.columns.values)
featureNames.remove('Risk')
x = df[featureNames]
y = df.Risk

neighbors = [3, 9, 15, 30]

def report(scores, experimentName):
	print(experimentName)
	print('Mean accuracy on train: %0.2f' % (scores['train_score'].mean()))
	print('Standard deviation accuracy on train: %0.2f' % (scores['train_score'].std()))
	print('Mean accuracy on test: %0.2f' % (scores['test_score'].mean()))
	print('Standard deviation accuracy on test: %0.2f' % (scores['test_score'].std()))

def expA(n):
	x_n = x
	experimento = '\n*** KNN - No Scaler - N={} - No weight***'.format(n)
	knn = KNeighborsClassifier(n_neighbors=n)
	#treinamento
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#resultados
	report(cv_scores, experimento)

for i in neighbors:
	expA(i)

def expB(n):
	x_n = x
	experimento = '\n*** KNN - No Scaler - N={} - Weight: inverse of distance***'.format(n)
	knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
	#treinamento
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#resultados
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
	#treinamento
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#resultados
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
	#treinamento
	cv_scores = cross_validate(knn, x_n, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
	#resultados
	report(cv_scores, experimento)

for i in neighbors:
	expD(i)
