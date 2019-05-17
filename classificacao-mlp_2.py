import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPClassifier
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

#normalização
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.transform(x)

print('MAX: ', x_n.max())
print('MIN: ', x_n.min())

iteracoes = 1500 #número de épocas
neuronios = 4
alpha = 0.04 #taxa de aprendizagem

log_file = open('mlp_final_out.txt', 'w+')

def report(scores):
	msg = 'Mean accuracy on train: %0.2f' % (scores['train_score'].mean())
	msg += '\nStandard deviation accuracy on train: %0.2f' % (scores['train_score'].std())
	msg += '\nMean accuracy on test: %0.2f' % (scores['test_score'].mean())
	msg += '\nStandard deviation accuracy on test: %0.2f' % (scores['test_score'].std())
	msg += '\nRun epochs on train: {}'.format(scores['train_epochs'])
	print(msg)
	log_file.write(msg + '\n')

def getRanEpochs(estimator, X, y):
	return estimator.n_iter_

def exp(i, n, a):
	experimento = '\n*** MLP - iteracoes={}; neuronios={}; alpha={} ***'.format(i, n, a)
	log_file.write(experimento + '\n')
	print(experimento)
	mlp = MLPClassifier(hidden_layer_sizes=(n), learning_rate_init=a, max_iter=i, n_iter_no_change=i, activation='tanh', solver='sgd', momentum=0.8)
	#treinamento
	cv_scores = cross_validate(mlp, x_n, y, scoring={'score':'accuracy', 'epochs':getRanEpochs}, cv=KFold(n_splits=10), return_train_score=True)
	#resultados
	report(cv_scores)

for i in range(5):
	exp(iteracoes, neuronios, alpha)

log_file.close()
