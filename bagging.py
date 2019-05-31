import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

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

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.transform(x)

def report(experimentName, scores):
	s = pd.Series(scores)
	print(experimentName)
	print('Mean on test: %0.2f' % (s.mean()))
	print('Standard deviation on test: %0.2f' % (s.std()))

def runExperiment(base_classifier, experimentName, scaleInput=False):
	max_samples = 0.6
	metric = 'precision'
	input = x
	if scaleInput:
		input = x_n
	for i in [10, 15, 20]:
		experiment = '\n*** Bagging - {} - {} base classifiers ***'.format(experimentName, i)
		test_scores = []
		for j in range(3):
			bagging = BaggingClassifier(base_classifier, n_estimators=i, max_samples=0.6)
			cv_scores = cross_validate(bagging, input, y, scoring=metric, cv=KFold(n_splits=10))
			test_scores.append(cv_scores['test_score'].mean())
		report(experiment, test_scores)

runExperiment(KNeighborsClassifier(n_neighbors=3), "KNN")

runExperiment(tree.DecisionTreeClassifier(), "Decision Tree")

runExperiment(GaussianNB(), "Naïve Bayes")

mlp = MLPClassifier(hidden_layer_sizes=(10), learning_rate_init=0.4, max_iter=1500, activation='tanh', solver='sgd', momentum=0.8)
runExperiment(mlp, "MLP", scaleInput=True)

