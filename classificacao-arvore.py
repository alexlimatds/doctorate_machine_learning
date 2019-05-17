import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn import tree

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Duas classes: 0 (No risk - baixo risco de fraude) e 1 (Risk - alto risco de fraude)
#471 instâncias da classe 0
#305 instâncias da classe 1

#Scikit Learn usa o algoritmo CART na sua implementação de árvore de decisão. Detalhes em: https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart

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

def report(cv_scores, experiment):
	print(experiment)
	print('Mean accuracy on train: %0.2f' % (cv_scores['train_score'].mean()))
	print('Standard deviation accuracy on train: %0.2f' % (cv_scores['train_score'].std()))
	print('Mean accuracy on test: %0.2f' % (cv_scores['test_score'].mean()))
	print('Standard deviation accuracy on test: %0.2f' % (cv_scores['test_score'].std()))
	print('Depth: %d' % (cv_scores['train_depth'].max()))

def getDepth(estimator, X, y):
	return estimator.tree_.max_depth

def exp(maxDepth):
	experiment = '\n*** Tree classifier - Max Depth {} ***'.format(maxDepth)
	classificador = tree.DecisionTreeClassifier(max_depth=maxDepth, random_state=0)
	#treinamento
	cv_scores = cross_validate(classificador, x, y, scoring={'score':'accuracy', 'depth':getDepth}, cv=KFold(n_splits=10), return_train_score=True)
	#resultados
	report(cv_scores, experiment)

depths = [None, 10, 5, 3, 2, 1]

for d in depths:
	exp(d)
