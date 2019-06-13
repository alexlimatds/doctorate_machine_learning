import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn import tree

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Two classes: 0 (No risk) e 1 (Risk)
#471 instances from 0 class
#305 instances frmo 1 class

#Scikit Learn uses CART algorithm in its decision tree implementation. Details at: https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart

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

log_file = open('classification_scores-decision_tree.txt', 'w+')

def report(cv_scores, experiment):
	print(experiment)
	print('Mean accuracy on train: %0.2f' % (cv_scores['train_score'].mean()))
	print('Standard deviation accuracy on train: %0.2f' % (cv_scores['train_score'].std()))
	print('Mean accuracy on test: %0.2f' % (cv_scores['test_score'].mean()))
	print('Standard deviation accuracy on test: %0.2f' % (cv_scores['test_score'].std()))
	print('Depth: %d' % (cv_scores['train_depth'].max()))
	#writing test scores
	log_file.write('{} score per fold\n'.format(experiment))
	for s in cv_scores['test_score']:
		log_file.write('{}\n'.format(s))

def getDepth(estimator, X, y):
	return estimator.tree_.max_depth

def exp(maxDepth):
	experiment = '\n*** Tree classifier - Max Depth {} ***'.format(maxDepth)
	classificador = tree.DecisionTreeClassifier(max_depth=maxDepth, random_state=0)
	#trainning
	cv_scores = cross_validate(classificador, x, y, scoring={'score':'accuracy', 'depth':getDepth}, cv=KFold(n_splits=10), return_train_score=True)
	#results
	report(cv_scores, experiment)

depths = [None, 10, 5, 3, 2, 1]

for d in depths:
	exp(d)

log_file.close()
