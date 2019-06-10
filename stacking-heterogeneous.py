import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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

featuresControl = 1

def report(experimentName, scores):
	s = pd.Series(scores)
	print(experimentName)
	print('Mean on test: %0.2f' % (s.mean()))
	print('Standard deviation on test: %0.3f' % (s.std()))

def generateMLP():
	return MLPClassifier(hidden_layer_sizes=(random.randint(5, 15)), learning_rate_init=random.random(), max_iter=random.randint(100, 2000), activation='tanh', solver='sgd', momentum=random.random())

def generateNaive():
	n_cols = random.randint(2, 6)
	cols = random.sample([0, 1, 2, 3, 4, 5, 6, 7], n_cols)
	return make_pipeline(ColumnSelector(cols=cols), GaussianNB())

def generateDT():
	criterion = random.sample(['entropy', 'gini'], 1)[0]
	max_features = random.sample(['sqrt', 'log2', None], 1)[0]
	splitter = random.sample(['best', 'random'], 1)[0]
	return tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=random.random(), max_features=max_features)

def getRandomFeatures():
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

def runExperiment(base_classifiers, experimentName, featureSelection=False):
	metric = 'precision'
	meta_classifier = GaussianNB()
	input = x
	for i in [10, 15, 20]:
		list_classifiers = []
		n = i // len(base_classifiers)
		for j in range(len(base_classifiers)):
			for k in range(n):
				if base_classifiers[j] == 'NB':
					if featureSelection:
						c = GaussianNB()
					else:
						c = generateNaive()
				elif base_classifiers[j] == 'MLP':
					c = generateMLP()
				elif base_classifiers[j] == 'DT':
					c = generateDT()
				else:
					raise ValueError('Base classifier not identified: {}'.format(base_classifiers[j]))
				if featureSelection:
					cols = getRandomFeatures()
					pipe = make_pipeline(ColumnSelector(cols=cols), c)
					list_classifiers.append(pipe)
				else: 				
					list_classifiers.append(c)
		test_scores = []
		experiment = '\n*** Stacking - {} - {} base classifiers ***'.format(experimentName, len(list_classifiers))
		for j in range(10):
			ensemble = StackingClassifier(classifiers=list_classifiers, meta_classifier=meta_classifier)
			cv_scores = cross_validate(ensemble, input, y, scoring=metric, cv=KFold(n_splits=10))
			test_scores.append(cv_scores['test_score'].mean())
		report(experiment, test_scores)

#runExperiment(['DT', 'NB'], "Decision Tree / Naive Bayes")

#runExperiment(['DT', 'MLP'], "Decision Tree / MLP")

#runExperiment(['NB', 'MLP'], "Naive Bayes / MLP")

#runExperiment(['DT', 'NB', 'MLP'], "Decision Tree / Naive Bayes / MLP")

runExperiment(['MLP', 'DT', 'NB'], "Decision Tree / Naive Bayes / MLP / Feature selection", featureSelection=True)
