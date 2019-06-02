import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
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
	print('Standard deviation on test: %0.2f' % (s.std()))

def runExperiment(base_classifiers, experimentName, scaleInput=False):
	metric = 'precision'
	meta_classifier = GaussianNB()
	input = x
	for i in [10, 15, 20]:
		list_classifiers = []
		n = i // len(base_classifiers)
		for j in range(len(base_classifiers)):
			for k in range(n):
				list_classifiers.append(base_classifiers[j])
		test_scores = []
		experiment = '\n*** Stacking - {} - {} base classifiers ***'.format(experimentName, len(list_classifiers))
		for j in range(3):
			ensemble = StackingClassifier(classifiers=list_classifiers, meta_classifier=meta_classifier)
			cv_scores = cross_validate(ensemble, input, y, scoring=metric, cv=KFold(n_splits=10))
			test_scores.append(cv_scores['test_score'].mean())
		report(experiment, test_scores)

mlp = MLPClassifier(hidden_layer_sizes=(10), learning_rate_init=0.4, max_iter=1500, activation='tanh', solver='sgd', momentum=0.8)
dt = tree.DecisionTreeClassifier()
nb = GaussianNB()

#runExperiment([dt, nb], "Decision Tree / Naive Bayes")

#runExperiment([dt, mlp], "Decision Tree / MLP")

#runExperiment([nb, mlp], "Naive Bayes / MLP")

pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)), mlp)
pipe2 = make_pipeline(ColumnSelector(cols=(2, 5)), dt)
pipe3 = make_pipeline(ColumnSelector(cols=(5, 7)), nb)
runExperiment([pipe1, pipe2, pipe3], "Decision Tree / Naive Bayes / MLP / Feature selection")