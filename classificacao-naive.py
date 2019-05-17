import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Duas classes: 0 (No risk - baixo risco de fraude) e 1 (Risk - alto risco de fraude)
#471 instâncias da classe 0
#305 instâncias da classe 1

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

#geração de histogramas
#x.hist(bins=10, figsize=(9, 10))
#pl.savefig('histogramas')

#matriz de correlação
#sns.heatmap(x.corr(), annot=True).figure.savefig('corr.png')

def report(scores, experimentName):
	print(experimentName)
	print('Mean accuracy on train: %0.2f' % (scores['train_score'].mean()))
	print('Standard deviation accuracy on train: %0.2f' % (scores['train_score'].std()))
	print('Mean accuracy on test: %0.2f' % (scores['test_score'].mean()))
	print('Standard deviation accuracy on test: %0.2f' % (scores['test_score'].std()))

naive = GaussianNB()
experimento = '*** NAIVE BAYES - No Scaler ***'
x_n = x
#treinamento
cv_scores = cross_validate(naive, x, y, scoring='accuracy', cv=KFold(n_splits=10), return_train_score=True)
#resultados
report(cv_scores, experimento)

