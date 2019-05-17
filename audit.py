import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

#divisão dos conjuntos de treinamento e de teste
featureNames = list(df.columns.values)
featureNames.remove('Risk')
x = df[featureNames]
y = df.Risk
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

def report(classifier, experimentName, train, test, train_target, test_target):
	print(experimentName)
	print('Accuracy of classifier on training set: {:.2f}'.format(classifier.score(train, train_target)))
	print('Accuracy of classifier on test set: {:.2f}'.format(classifier.score(test, test_target)))
	print('Confusion matrix of test set:')
	y_pred = classifier.predict(test)
	cm = confusion_matrix(test_target, y_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print(cm)

experimento = '*** NAIVE BAYES - StandardScaler ***'
naive = GaussianNB()
#normalização
scaler = StandardScaler()
scaler.fit(x)
x_n_train = scaler.transform(x_train)
x_n_test = scaler.transform(x_test)
#treinamento
naive.fit(x_n_train, y_train)
#resultados
report(naive, experimento, x_n_train, x_n_test, y_train, y_test)

experimento = '*** NAIVE BAYES - MinMaxScaler ***'
#normalização
scaler = MinMaxScaler()
scaler.fit(x)
x_n_train = scaler.transform(x_train)
x_n_test = scaler.transform(x_test)
#treinamento
naive.fit(x_n_train, y_train)
#resultados
report(naive, experimento, x_n_train, x_n_test, y_train, y_test)

experimento = '*** NAIVE BAYES - RobustScaler ***'
#normalização
scaler = RobustScaler()
scaler.fit(x)
x_n_train = scaler.transform(x_train)
x_n_test = scaler.transform(x_test)
#treinamento
naive.fit(x_n_train, y_train)
#resultados
report(naive, experimento, x_n_train, x_n_test, y_train, y_test)

experimento = '*** NAIVE BAYES - Normalizer ***'
#normalização
scaler = Normalizer()
scaler.fit(x)
x_n_train = scaler.transform(x_train)
x_n_test = scaler.transform(x_test)
#treinamento
naive.fit(x_n_train, y_train)
#resultados
report(naive, experimento, x_n_train, x_n_test, y_train, y_test)

experimento = '*** KNN - 5 ***'
n_neighbors = 5;
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#normalização
scaler = StandardScaler()
scaler.fit(x)
x_n_train = scaler.transform(x_train)
x_n_test = scaler.transform(x_test)
#treinamento
knn.fit(x_n_train, y_train)
#resultados
report(knn, experimento, x_n_train, x_n_test, y_train, y_test)

experimento = '*** MLP ***'
mlp = MLPClassifier(hidden_layer_sizes=(10), activation='tanh', alpha=0.01)
#normalização
scaler = StandardScaler()
scaler.fit(x)
x_n_train = scaler.transform(x_train)
x_n_test = scaler.transform(x_test)
#treinamento
mlp.fit(x_n_train, y_train)
#resultados
report(mlp, experimento, x_n_train, x_n_test, y_train, y_test)
