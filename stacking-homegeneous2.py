import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

#Extracting columns that will be used
cols = ['PARA_A', 'PARA_B', 'Money_Value', 'numbers', 'Sector_score', 'District', 'History_score', 'LOSS_SCORE', 'Risk']
df = df[cols]

#Removing instances with empty columns
df = df.dropna(subset=['Money_Value'])

#Extracting input vectors and classes
featureNames = list(df.columns.values)
featureNames.remove('Risk')
x = df[featureNames]
y = df.Risk

def diversity(estimator, X, y):
	N = len(y.index)
	classifiers = estimator.clfs_
	L = len(classifiers)
	sum = 0
	out = []
	y2 = y.to_list()
	for c in classifiers:
		out.append(c.predict(X))
	for i in range(0, N):	
		l = 0
		for o in out:
			if o[i] == y2[i]:
				l = l + 1
		sum = sum + (1 / (L - (L/ 2)) * min(l, L - l))
	E = sum / N
	return E;

def report(experimentName, scores):
	score = pd.Series(scores['score'])
	diversity = pd.Series(scores['diversity'])
	print(experimentName)
	print('Mean score on test: %0.2f' % (score.mean()))
	print('Standard deviation score on test: %0.3f' % (score.std()))
	print('Mean diversity on test: %0.4f' % (diversity.mean()))
	print('Standard deviation diversity on test: %0.4f' % (diversity.std()))

featuresControl = 1
def getRandomCols():
	global featuresControl
	i = featuresControl % 4
	featuresControl = featuresControl + 1
	if i == 1:
		return (0, 1, 2, 3)
	elif i == 2:
		return (4, 5, 6, 7)
	elif i == 3:
		return (0, 1, 6, 7)
	else:
		return (2, 3, 4, 5)

def runExperiment(bases, experimentName):
	metric = 'precision'
	meta_classifier = GaussianNB()
	input = x
	for i in [10, 15, 20]:
		experiment = '\n*** Stacking - {} - {} base classifiers ***'.format(experimentName, i)
		base_classifiers = []
		while len(base_classifiers) < i:
			for b in bases:
				pipe = make_pipeline(ColumnSelector(cols=getRandomCols()), b)
				base_classifiers.append(pipe)
#		test_scores = []
		test_scores = {'score':[], 'diversity':[]}
		for j in range(10):
			ensemble = StackingClassifier(classifiers=base_classifiers, meta_classifier=meta_classifier)
			cv_scores = cross_validate(ensemble, input, y, scoring={'score':metric, 'diversity':diversity}, cv=KFold(n_splits=10))
#			test_scores.append(cv_scores['test_score'].mean())
			test_scores['score'].append(cv_scores['test_score'].mean())
			test_scores['diversity'].append(cv_scores['test_diversity'].mean())
		report(experiment, test_scores)

bases = [KNeighborsClassifier(n_neighbors=1), 
	KNeighborsClassifier(n_neighbors=3), 
	KNeighborsClassifier(n_neighbors=10), 
	KNeighborsClassifier(n_neighbors=5, weights='distance'), 
	KNeighborsClassifier(n_neighbors=15, weights='distance')]
runExperiment(bases, 'KNN with feature selection')
