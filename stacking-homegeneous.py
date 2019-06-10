import pandas as pd
import numpy as np
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

def report(experimentName, scores):
	s = pd.Series(scores)
	print(experimentName)
	print('Mean on test: %0.2f' % (s.mean()))
	print('Standard deviation on test: %0.3f' % (s.std()))

def runExperiment(base_classifiers, experimentName):
	if len(base_classifiers) != 5:
		raise ValueError('You must provide exactly 5 base classifiers')
	metric = 'precision'
	meta_classifier = GaussianNB()
	input = x
	for i in [10, 15, 20]:
		classifiers = []
		for j in range(1, i // len(base_classifiers) + 1):
			classifiers.extend(base_classifiers)
		experiment = '\n*** Stacking - {} - {} base classifiers ***'.format(experimentName, len(classifiers))
		test_scores = []
		for j in range(10):
			ensemble = StackingClassifier(classifiers=base_classifiers, meta_classifier=meta_classifier)
			cv_scores = cross_validate(ensemble, input, y, scoring=metric, cv=KFold(n_splits=10))
			test_scores.append(cv_scores['test_score'].mean())
		report(experiment, test_scores)

"""
classifiers = [tree.DecisionTreeClassifier(), 
	tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=10), 
	tree.DecisionTreeClassifier(criterion='entropy', max_features='sqrt'), 
	tree.DecisionTreeClassifier(min_samples_split=10, splitter='random', max_features='log2'), 
	tree.DecisionTreeClassifier(splitter='random', max_features=0.7)]
runExperiment(classifiers, "Decision Tree")
"""
"""
classifiers = [KNeighborsClassifier(n_neighbors=1), 
	KNeighborsClassifier(n_neighbors=3), 
	KNeighborsClassifier(n_neighbors=10), 
	KNeighborsClassifier(n_neighbors=5, weights='distance'), 
	KNeighborsClassifier(n_neighbors=15, weights='distance')]
runExperiment(classifiers, "KNN")
"""
"""
classifiers =[make_pipeline(ColumnSelector(cols=(0, 1, 2)), GaussianNB()), 
	make_pipeline(ColumnSelector(cols=(2, 3, 4)), GaussianNB()), 
	make_pipeline(ColumnSelector(cols=(4, 5, 6)), GaussianNB()), 
	make_pipeline(ColumnSelector(cols=(5, 6, 7)), GaussianNB()), 
	make_pipeline(ColumnSelector(cols=(0, 3, 5, 7)), GaussianNB())]
runExperiment(classifiers, "Naive Bayes")
"""
classifiers = [MLPClassifier(hidden_layer_sizes=(10), learning_rate_init=0.4, max_iter=1500, activation='tanh', solver='sgd', momentum=0.8), 
	MLPClassifier(hidden_layer_sizes=(12), learning_rate_init=0.65, max_iter=1000, activation='tanh', solver='sgd', momentum=0.7), 
	MLPClassifier(hidden_layer_sizes=([5, 5]), learning_rate_init=0.8, max_iter=500, activation='tanh', solver='sgd', momentum=0.4), 
	MLPClassifier(hidden_layer_sizes=([3, 3, 3]), learning_rate_init=0.6, max_iter=200, activation='tanh', solver='sgd', momentum=0.3), 
	MLPClassifier(hidden_layer_sizes=(5), learning_rate_init=0.7, max_iter=200, activation='tanh', solver='sgd', momentum=0.9)]
runExperiment(classifiers, "MLP")

